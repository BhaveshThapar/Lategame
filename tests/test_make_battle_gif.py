"""Tests for the dependency-free replay -> GIF renderer.

The load-bearing test is `test_every_frame_round_trips_through_an_independent_decoder`. A malformed
LZW stream does not fail loudly -- it decodes to plausible noise partway through the animation, and
with no image library on this machine there is nothing to open the file with and check. So the
decoder here is written from the GIF spec independently of the encoder (different data structures,
opposite direction, its own bit reader) and the two are made to agree on real frames. Two
implementations agreeing is the evidence; neither alone is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_battle_gif import (  # noqa: E402
    DEFAULT_REPLAY,
    DEFAULT_TURNS,
    FONT,
    GLYPH_H,
    GLYPH_W,
    PALETTE,
    SIZE_BUDGET_BYTES,
    Canvas,
    _hp_fraction,
    _species,
    build_frames,
    build_gif,
    extract_log,
    lzw_encode,
    parse_meta,
    render,
    score_replay,
    write_gif,
)

COMMITTED_GIF = Path(__file__).resolve().parents[1] / "assets" / "rotomai_gen9ou.gif"
REPLAY = Path(__file__).resolve().parents[1] / DEFAULT_REPLAY

# `replays/` is gitignored, so a clone has no source HTML. The rendered asset IS committed, which is
# what CI checks.
requires_replay = pytest.mark.skipif(
    not REPLAY.exists(), reason="replays/ is gitignored; no saved replay on this machine"
)


# ---------------------------------------------------------------------------------------------
# An independent GIF reader, written from the spec rather than from the encoder.
# ---------------------------------------------------------------------------------------------


class _Bits:
    """LSB-first bit reader over a byte string."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read(self, width: int) -> int:
        while self.nbits < width:
            if self.pos >= len(self.data):
                raise EOFError("ran out of bits before the end-of-information code")
            self.acc |= self.data[self.pos] << self.nbits
            self.pos += 1
            self.nbits += 8
        value = self.acc & ((1 << width) - 1)
        self.acc >>= width
        self.nbits -= width
        return value


def lzw_decode(data: bytes, min_code_size: int) -> bytes:
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    bits = _Bits(data)
    out = bytearray()

    table: list[bytes] = []
    width = 0
    previous: bytes | None = None

    def reset() -> None:
        nonlocal table, width, previous
        table = [bytes([i]) for i in range(clear_code)] + [b"", b""]
        width = min_code_size + 1
        previous = None

    reset()
    while True:
        code = bits.read(width)
        if code == clear_code:
            reset()
            continue
        if code == end_code:
            return bytes(out)
        if code < len(table):
            entry = table[code]
        elif previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError(f"code {code} before any dictionary entry")
        out += entry
        if previous is not None:
            table.append(previous + entry[:1])
            if len(table) == (1 << width) and width < 12:
                width += 1
        previous = entry


def read_gif(path_or_bytes: Path | bytes) -> tuple[int, int, list[tuple[bytes, int]]]:
    """Return `(width, height, [(pixels, delay_cs), ...])` from a GIF89a file."""
    raw = path_or_bytes if isinstance(path_or_bytes, bytes) else path_or_bytes.read_bytes()
    assert raw[:6] == b"GIF89a", "not a GIF89a stream"
    width = raw[6] | (raw[7] << 8)
    height = raw[8] | (raw[9] << 8)
    packed = raw[10]
    assert packed & 0x80, "expected a global colour table"
    table_size = 1 << ((packed & 0x07) + 1)
    pos = 13 + 3 * table_size

    def sub_blocks(at: int) -> tuple[bytes, int]:
        chunks = bytearray()
        while raw[at] != 0:
            length = raw[at]
            chunks += raw[at + 1 : at + 1 + length]
            at += 1 + length
        return bytes(chunks), at + 1

    frames: list[tuple[bytes, int]] = []
    delay = 0
    while pos < len(raw):
        marker = raw[pos]
        if marker == 0x3B:  # trailer
            break
        if marker == 0x21:  # extension
            label = raw[pos + 1]
            if label == 0xF9:  # graphic control
                delay = raw[pos + 4] | (raw[pos + 5] << 8)
                pos += 3 + raw[pos + 2]
                assert raw[pos] == 0
                pos += 1
            else:
                pos += 2
                if label == 0xFF:  # application extension: one length-prefixed identifier first
                    pos += 1 + raw[pos]
                _, pos = sub_blocks(pos)
            continue
        if marker == 0x2C:  # image descriptor
            frame_w = raw[pos + 5] | (raw[pos + 6] << 8)
            frame_h = raw[pos + 7] | (raw[pos + 8] << 8)
            local = raw[pos + 9]
            assert not local & 0x80, "no local colour tables are written"
            pos += 10
            min_code_size = raw[pos]
            data, pos = sub_blocks(pos + 1)
            pixels = lzw_decode(data, min_code_size)
            assert len(pixels) == frame_w * frame_h, (
                f"decoded {len(pixels)} pixels, image descriptor says {frame_w * frame_h}"
            )
            frames.append((pixels, delay))
            continue
        raise ValueError(f"unexpected block marker {marker:#x} at {pos}")
    return width, height, frames


# ---------------------------------------------------------------------------------------------
# LZW
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00",
        b"\x00" * 4096,
        bytes(range(8)) * 700,
        bytes([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0] * 512),
    ],
    ids=["single", "flat", "cycling", "palindromic"],
)
def test_lzw_round_trips(payload: bytes) -> None:
    assert lzw_decode(lzw_encode(payload, 3), 3) == payload


def test_lzw_round_trips_past_the_dictionary_limit() -> None:
    """Forces the code width past 12 bits and through a mid-stream table reset.

    A pseudo-random byte stream fills the 4096-entry dictionary, which is the only path that
    exercises the encoder's `Clear`-and-restart branch -- the branch a short test never reaches and
    the one an off-by-one in the width rule corrupts.
    """
    state = 1
    data = bytearray()
    for _ in range(200_000):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        data.append((state >> 16) & 0x07)
    payload = bytes(data)
    assert lzw_decode(lzw_encode(payload, 3), 3) == payload


def test_the_code_width_never_exceeds_twelve_bits() -> None:
    """A 13-bit code is unreadable by every conforming decoder; the encoder must Clear instead."""
    state = 7
    data = bytearray()
    for _ in range(300_000):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        data.append((state >> 16) & 0x07)
    encoded = lzw_encode(bytes(data), 3)
    assert lzw_decode(encoded, 3) == bytes(data)


# ---------------------------------------------------------------------------------------------
# Canvas + font
# ---------------------------------------------------------------------------------------------


def test_every_glyph_is_five_by_seven() -> None:
    for char, rows in FONT.items():
        assert len(rows) == GLYPH_H, char
        assert all(0 <= r < (1 << GLYPH_W) for r in rows), char


def test_the_font_covers_what_the_captions_need() -> None:
    needed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:-!?'()%/+_>")
    assert needed <= set(FONT)


def test_canvas_rect_clips_rather_than_wrapping() -> None:
    canvas = Canvas(10, 4)
    canvas.rect(-3, -3, 5, 5, 1)
    canvas.rect(8, 2, 40, 40, 2)
    assert canvas.px[0] == 1
    assert canvas.px[2] == 0, "a negative-origin rect must not wrap into the next column"
    assert canvas.px[3 * 10 + 9] == 2
    assert len(canvas.px) == 40, "clipping must not resize the buffer"


def test_text_draws_inside_its_advance() -> None:
    canvas = Canvas(64, 16)
    end = canvas.text(2, 2, "A", 1, scale=2)
    assert end == 2 + (GLYPH_W + 1) * 2
    assert any(canvas.px[y * 64 + x] == 1 for y in range(2, 16) for x in range(2, end))


def test_an_unknown_character_falls_back_rather_than_raising() -> None:
    canvas = Canvas(32, 16)
    canvas.text(0, 0, "☃", 1)  # snowman
    assert 1 in canvas.px


# ---------------------------------------------------------------------------------------------
# GIF container
# ---------------------------------------------------------------------------------------------


def test_a_written_gif_has_the_expected_structure(tmp_path: Path) -> None:
    canvas = Canvas(8, 4, fill=0)
    canvas.rect(1, 1, 3, 2, 5)
    out = tmp_path / "t.gif"
    size = write_gif(out, [canvas, canvas], [10, 10])
    raw = out.read_bytes()
    assert size == len(raw)
    assert raw.startswith(b"GIF89a")
    assert raw.endswith(b"\x3b")
    assert b"NETSCAPE2.0" in raw, "without the loop extension the animation plays once"
    width, height, frames = read_gif(out)
    assert (width, height) == (8, 4)
    assert len(frames) == 2
    assert frames[0][0] == bytes(canvas.px)


def test_frames_must_agree_on_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_gif(tmp_path / "t.gif", [Canvas(4, 4), Canvas(8, 4)], [10, 10])


def test_a_delay_is_required_per_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_gif(tmp_path / "t.gif", [Canvas(4, 4), Canvas(4, 4)], [10])


def test_the_palette_fits_the_declared_colour_depth(tmp_path: Path) -> None:
    out = tmp_path / "t.gif"
    write_gif(out, [Canvas(4, 4)], [10])
    raw = out.read_bytes()
    bits = (raw[10] & 0x07) + 1
    assert (1 << bits) >= len(PALETTE)


# ---------------------------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------------------------


def test_extract_log_rejects_a_page_that_is_not_a_replay() -> None:
    with pytest.raises(ValueError):
        extract_log("<html><body>nope</body></html>")


def test_extract_log_unescapes_entities() -> None:
    page = '<script class="battle-log-data">|title|a &amp; b</script>'
    assert extract_log(page) == "|title|a & b"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("100/100", 1.0),
        ("85/100", 0.85),
        ("0 fnt", 0.0),
        ("309/341", 309 / 341),
        ("50/100 par", 0.5),
        ("", 0.0),
    ],
)
def test_hp_fraction(token: str, expected: float) -> None:
    assert _hp_fraction(token) == pytest.approx(expected)


def test_species_prefers_the_detail_field_over_the_nickname() -> None:
    assert _species("p1a: Fluffy", "Abomasnow, F, tera:Grass") == "Abomasnow"
    assert _species("p1a: Dragapult") == "Dragapult"


SYNTHETIC = "\n".join(
    [
        "|player|p1|human|170|1004",
        "|player|p2|RotomLover12|266|1000",
        "|tier|[Gen 9] OU",
        "|rated|",
        "|turn|1",
        "|switch|p1a: Jirachi|Jirachi|100/100",
        "|switch|p2a: Kingambit|Kingambit|341/341",
        "|turn|2",
        "|move|p2a: Kingambit|Sucker Punch|p1a: Jirachi",
        "|-supereffective|p1a: Jirachi",
        "|-damage|p1a: Jirachi|0 fnt",
        "|faint|p1a: Jirachi",
        "|win|RotomLover12",
    ]
)


def test_parse_meta_reads_players_ratings_and_tier() -> None:
    meta = parse_meta(SYNTHETIC)
    assert (meta.p1, meta.p2) == ("human", "RotomLover12")
    assert meta.rating == {"p1": "1004", "p2": "1000"}
    assert meta.tier == "[Gen 9] OU"
    assert meta.rated is True
    assert meta.winner == "RotomLover12"


def test_frames_are_confined_to_the_requested_turn_window() -> None:
    meta = parse_meta(SYNTHETIC)
    frames = build_frames(SYNTHETIC, meta, "RotomLover12", 2, 2)
    assert frames, "the window must produce frames"
    assert {f.turn for f in frames} == {2}


def test_state_outside_the_window_still_applies() -> None:
    """Turn 2's frames must know what turn 1 switched in, or the roster renders empty."""
    meta = parse_meta(SYNTHETIC)
    frames = build_frames(SYNTHETIC, meta, "RotomLover12", 2, 2)
    assert frames[0].p1.active == "Jirachi"
    assert frames[0].p2.active == "Kingambit"


def test_a_faint_increments_the_ko_count_and_zeroes_hp() -> None:
    meta = parse_meta(SYNTHETIC)
    frames = build_frames(SYNTHETIC, meta, "RotomLover12", 1, 2)
    faint = next(f for f in frames if "FAINTED" in f.caption)
    assert faint.p1.fainted == 1
    assert faint.p1.hp == 0.0
    assert faint.detail == "KO 1-0"


def test_super_effective_is_carried_onto_the_damage_frame() -> None:
    meta = parse_meta(SYNTHETIC)
    frames = build_frames(SYNTHETIC, meta, "RotomLover12", 2, 2)
    damage = next(f for f in frames if "->" in f.caption)
    assert damage.detail == "SUPER EFFECTIVE!"


def test_recovery_updates_the_bar_without_emitting_a_frame() -> None:
    """Leftovers is not a beat, but ignoring it freezes the bar at the last hit taken."""
    log = SYNTHETIC.replace(
        "|faint|p1a: Jirachi",
        "|faint|p1a: Jirachi\n|-heal|p2a: Kingambit|330/341|[from] item: Leftovers",
    )
    meta = parse_meta(log)
    frames = build_frames(log, meta, "RotomLover12", 1, 2)
    assert not any("HEAL" in f.caption for f in frames)
    assert frames[-1].p2.hp == pytest.approx(330 / 341)


@requires_replay
def test_the_final_frame_matches_the_battles_own_endgame() -> None:
    """The agent closes on 220/341 with a 6-3 KO margin; a frozen bar would read 178/341."""
    log = extract_log(REPLAY.read_text(encoding="utf-8", errors="replace"))
    meta = parse_meta(log)
    final = build_frames(log, meta, "RotomLover12", *DEFAULT_TURNS)[-1]
    assert (final.p1.fainted, final.p2.fainted) == (6, 3)
    assert final.p2.hp == pytest.approx(220 / 341)
    assert final.p1.hp == 0.0


def test_the_agent_must_have_played_the_battle(tmp_path: Path) -> None:
    page = tmp_path / "r.html"
    page.write_text(f'<script class="battle-log-data">{SYNTHETIC}</script>', encoding="utf-8")
    with pytest.raises(ValueError, match="did not play"):
        build_gif(page, tmp_path / "o.gif", us="SomeoneElse")


def test_an_empty_turn_window_is_an_error_rather_than_an_empty_gif(tmp_path: Path) -> None:
    page = tmp_path / "r.html"
    page.write_text(f'<script class="battle-log-data">{SYNTHETIC}</script>', encoding="utf-8")
    with pytest.raises(ValueError, match="no events"):
        build_gif(page, tmp_path / "o.gif", us="RotomLover12", turns=(40, 50))


def test_score_replay_counts_both_sides() -> None:
    meta = parse_meta(SYNTHETIC)
    turns, ours, theirs, won = score_replay(SYNTHETIC, meta, "RotomLover12")
    assert (turns, ours, theirs, won) == (2, 1, 0, True)


def test_render_produces_a_full_canvas_of_palette_indices() -> None:
    meta = parse_meta(SYNTHETIC)
    frames = build_frames(SYNTHETIC, meta, "RotomLover12", 1, 2)
    canvas = render(frames[-1], meta, "RotomLover12", 1, 2)
    assert len(canvas.px) == canvas.width * canvas.height
    assert set(canvas.px) <= set(range(len(PALETTE)))
    assert len(set(canvas.px)) > 2, "a rendered frame should use more than background and text"


def test_a_synthetic_battle_round_trips_end_to_end(tmp_path: Path) -> None:
    page = tmp_path / "r.html"
    page.write_text(f'<script class="battle-log-data">{SYNTHETIC}</script>', encoding="utf-8")
    out = tmp_path / "o.gif"
    count, size = build_gif(page, out, us="RotomLover12", turns=(1, 2))
    assert count > 0
    _, _, frames = read_gif(out)
    assert len(frames) == count
    assert size <= SIZE_BUDGET_BYTES


# ---------------------------------------------------------------------------------------------
# The committed asset, and the real replay when it is present
# ---------------------------------------------------------------------------------------------


def test_the_committed_gif_decodes_and_fits_the_budget() -> None:
    assert COMMITTED_GIF.exists(), "the README's first screen links this file"
    assert COMMITTED_GIF.stat().st_size <= SIZE_BUDGET_BYTES
    width, height, frames = read_gif(COMMITTED_GIF)
    assert (width, height) == (560, 260)
    assert len(frames) >= 10, "an animation, not a still"
    assert all(len(px) == width * height for px, _ in frames)
    assert sum(delay for _, delay in frames) >= 300, "under 3s total is too fast to read"


@requires_replay
def test_every_frame_round_trips_through_an_independent_decoder(tmp_path: Path) -> None:
    log = extract_log(REPLAY.read_text(encoding="utf-8", errors="replace"))
    meta = parse_meta(log)
    frames = build_frames(log, meta, "RotomLover12", *DEFAULT_TURNS)
    canvases = [render(f, meta, "RotomLover12", *DEFAULT_TURNS) for f in frames]
    out = tmp_path / "o.gif"
    write_gif(out, canvases, [f.delay_cs for f in frames])
    _, _, decoded = read_gif(out)
    assert len(decoded) == len(canvases)
    for i, (pixels, _) in enumerate(decoded):
        assert pixels == bytes(canvases[i].px), f"frame {i} did not survive the encoder"


@requires_replay
def test_the_selected_battle_is_the_one_operations_describes() -> None:
    log = extract_log(REPLAY.read_text(encoding="utf-8", errors="replace"))
    meta = parse_meta(log)
    turns, ours, theirs, won = score_replay(log, meta, "RotomLover12")
    assert (turns, ours, theirs, won) == (17, 6, 3, True)


@requires_replay
def test_the_committed_gif_is_reproducible_from_the_replay(tmp_path: Path) -> None:
    """Regenerating must be byte-identical, or the committed asset has drifted from the code."""
    out = tmp_path / "o.gif"
    build_gif(REPLAY, out, us="RotomLover12", turns=DEFAULT_TURNS)
    assert out.read_bytes() == COMMITTED_GIF.read_bytes()
