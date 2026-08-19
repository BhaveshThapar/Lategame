"""Render a saved Showdown replay's endgame as an animated GIF, with no third-party deps.

`docs/OPERATIONS.md` describes the sprite path: open the saved `.html`, screen-record it, and run
two-pass `ffmpeg`. That needs a browser with network (the page pulls `replay-embed.js` from
play.pokemonshowdown.com), plus `ffmpeg`. The machine this project runs on has none of those and no
egress, so the sprite GIF could not be produced here -- and a README whose first screen is a broken
image link is worse than one with no image at all.

**This renders the LOG, not the sprites.** Every `.html` poke-env saves embeds the complete battle
log in a `battle-log-data` script tag, and the log is the authoritative record of what happened --
the sprites are a rendering of it. So this walks the log and draws each event directly: rosters, the
active Pokemon per side, HP bars, the KO count, and a caption naming the move. It is honest about
being a log replay and it costs no dependency, no browser and no network.

Nothing here imports anything outside the standard library, deliberately. Pillow is not installed
and adding an imaging dependency to the project for one committed asset is a bad trade, so the 5x7
font, the rasteriser and the GIF89a/LZW encoder are all in this file.

    python scripts/make_battle_gif.py                       # the battle OPERATIONS.md selected
    python scripts/make_battle_gif.py --replay X --out Y     # any other saved replay
    python scripts/make_battle_gif.py --list                 # rank saved replays by watchability
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The battle `docs/OPERATIONS.md` picked from the first ladder block: a 6-3 win in 17 turns against
# a 1004-Elo human, closing with three consecutive Sucker Punch KOs from turn 13.
DEFAULT_REPLAY = Path("replays/live_ladder_gen9ou/RotomLover12 - battle-gen9ou-2666751310.html")
DEFAULT_OUT = Path("assets/rotomai_gen9ou.gif")
DEFAULT_US = "RotomLover12"
DEFAULT_TURNS = (13, 17)

#: The README budget for a first-screen asset (`docs/OPERATIONS.md`).
SIZE_BUDGET_BYTES = 3 * 1024 * 1024


# --------------------------------------------------------------------------------------------
# Log extraction
# --------------------------------------------------------------------------------------------

_LOG_RE = re.compile(
    r'<script[^>]*class="battle-log-data"[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE
)
_LOG_RE_LOOSE = re.compile(r"<script[^>]*battle-log-data[^>]*>(.*?)</script>", re.DOTALL)


def extract_log(html_text: str) -> str:
    """Pull the raw `|`-delimited battle log out of a poke-env-saved replay page.

    The log is HTML-escaped inside the script tag (`&amp;`, `&lt;` in nicknames and chat), so it is
    unescaped here rather than at every use site.
    """
    match = _LOG_RE.search(html_text) or _LOG_RE_LOOSE.search(html_text)
    if match is None:
        raise ValueError("no `battle-log-data` script tag -- is this a saved Showdown replay?")
    return html.unescape(match.group(1))


# --------------------------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------------------------


@dataclass
class Meta:
    """Who played, and under what."""

    p1: str = "p1"
    p2: str = "p2"
    rating: dict[str, str] = field(default_factory=dict)
    tier: str = ""
    rated: bool = False
    winner: str = ""


@dataclass
class Side:
    """One player's visible state as the log reveals it."""

    active: str = ""
    hp: float = 1.0
    fainted: int = 0


def _hp_fraction(token: str) -> float:
    """`85/100`, `309/341`, `0 fnt`, `100/100 par` -> a fraction in [0, 1].

    Two scales appear in one log and must not be mixed: HP Percentage Mod shows the *opponent* in
    percent while the POV player's own side keeps real HP, so nothing may compare the raw numerator
    across sides.
    """
    head = token.split(" ")[0].strip()
    if not head or head == "0":
        return 0.0
    if "/" not in head:
        return 0.0
    num, _, den = head.partition("/")
    try:
        cur, total = float(num), float(den)
    except ValueError:
        return 0.0
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, cur / total))


def _species(ident: str, detail: str = "") -> str:
    """`p1a: Iron Valiant` + `Iron Valiant, F, tera:Fairy` -> the species, nickname discarded."""
    if detail:
        return detail.split(",")[0].strip()
    _, _, name = ident.partition(": ")
    return name.strip() or ident


def _side_of(ident: str) -> str:
    """`p2a: Kingambit` -> `p2`."""
    return ident[:2]


def parse_meta(log: str) -> Meta:
    """Read the pre-battle header: players, ratings, tier, and the eventual winner."""
    meta = Meta()
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "player" and len(parts) >= 4 and parts[3]:
            if parts[2] == "p1":
                meta.p1 = parts[3]
            elif parts[2] == "p2":
                meta.p2 = parts[3]
            if len(parts) >= 6 and parts[5].strip():
                meta.rating[parts[2]] = parts[5].strip()
        elif tag == "tier":
            meta.tier = parts[2] if len(parts) > 2 else ""
        elif tag == "rated":
            meta.rated = True
        elif tag == "win" and len(parts) > 2:
            meta.winner = parts[2]
    return meta


@dataclass
class Frame:
    """One rendered beat: the state to draw plus the caption explaining it."""

    turn: int
    p1: Side
    p2: Side
    caption: str
    detail: str
    delay_cs: int


def _pct(hp: float) -> str:
    return f"{round(hp * 100):d}%"


def build_frames(
    log: str, meta: Meta, us: str, first_turn: int, last_turn: int
) -> list[Frame]:
    """Walk the log and emit one frame per watchable beat inside `[first_turn, last_turn]`.

    The whole log is walked, not just the selected window: HP and roster state at turn 13 is only
    correct if turns 1-12 were applied. Frames are emitted only inside the window.
    """
    sides = {"p1": Side(), "p2": Side()}
    us_side = "p2" if meta.p2 == us else "p1"
    them_side = "p1" if us_side == "p2" else "p2"
    label = {us_side: "ROTOMAI", them_side: meta.p1 if them_side == "p1" else meta.p2}

    frames: list[Frame] = []
    turn = 0
    pending_effect = ""

    def emit(caption: str, detail: str, delay: int) -> None:
        if first_turn <= turn <= last_turn:
            frames.append(
                Frame(
                    turn=turn,
                    p1=Side(sides["p1"].active, sides["p1"].hp, sides["p1"].fainted),
                    p2=Side(sides["p2"].active, sides["p2"].hp, sides["p2"].fainted),
                    caption=caption,
                    detail=detail,
                    delay_cs=delay,
                )
            )

    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == "turn":
            turn = int(parts[2])
            if first_turn <= turn <= last_turn:
                emit(f"TURN {turn}", "", 110)
            pending_effect = ""

        elif tag in ("switch", "drag") and len(parts) >= 5:
            side = _side_of(parts[2])
            sides[side].active = _species(parts[2], parts[3])
            sides[side].hp = _hp_fraction(parts[4])
            emit(
                f"{label[side]} SENT OUT {sides[side].active.upper()}",
                _pct(sides[side].hp),
                110,
            )

        elif tag == "move" and len(parts) >= 4:
            side = _side_of(parts[2])
            emit(
                f"{sides[side].active.upper()} USED {parts[3].upper()}",
                label[side],
                110,
            )

        elif tag in ("-supereffective", "-crit"):
            pending_effect = (
                "SUPER EFFECTIVE!" if tag == "-supereffective" else "CRITICAL HIT!"
            )

        elif tag == "-damage" and len(parts) >= 4:
            side = _side_of(parts[2])
            before = sides[side].hp
            sides[side].hp = _hp_fraction(parts[3])
            emit(
                f"{sides[side].active.upper()} {_pct(before)} -> {_pct(sides[side].hp)}",
                pending_effect,
                110,
            )
            pending_effect = ""

        elif tag in ("-heal", "-sethp") and len(parts) >= 4:
            # State only, no frame. Leftovers ticking every `upkeep` is not a beat worth a frame,
            # but dropping it entirely leaves the bar frozen at the last hit -- the agent's
            # Kingambit reads 52% for the rest of the game instead of recovering to 65%.
            sides[_side_of(parts[2])].hp = _hp_fraction(parts[3])

        elif tag == "faint" and len(parts) >= 3:
            side = _side_of(parts[2])
            sides[side].hp = 0.0
            sides[side].fainted += 1
            # The scoreboard is always stated from our side: KOs we scored first. `fainted` counts
            # losses, so ours is THEIR fainted count -- inverting this reads as a losing position
            # in a battle the agent won.
            emit(
                f"{sides[side].active.upper()} FAINTED",
                f"KO {sides[them_side].fainted}-{sides[us_side].fainted}",
                160,
            )

        elif tag == "win" and len(parts) >= 3:
            winner_side = us_side if parts[2] == us else them_side
            emit(
                f"{label[winner_side]} WINS",
                f"{sides[them_side].fainted}-{sides[us_side].fainted} ON KOS",
                400,
            )

    return frames


# --------------------------------------------------------------------------------------------
# Font: 5x7, uppercase-only, drawn rather than tabulated so it can be read and corrected.
# --------------------------------------------------------------------------------------------

GLYPH_W, GLYPH_H = 5, 7

_FONT_SRC: dict[str, str] = {
    "A": ".###./#...#/#...#/#####/#...#/#...#/#...#",
    "B": "####./#...#/#...#/####./#...#/#...#/####.",
    "C": ".###./#...#/#..../#..../#..../#...#/.###.",
    "D": "####./#...#/#...#/#...#/#...#/#...#/####.",
    "E": "#####/#..../#..../####./#..../#..../#####",
    "F": "#####/#..../#..../####./#..../#..../#....",
    "G": ".###./#...#/#..../#.###/#...#/#...#/.###.",
    "H": "#...#/#...#/#...#/#####/#...#/#...#/#...#",
    "I": ".###./..#../..#../..#../..#../..#../.###.",
    "J": "..###/...#./...#./...#./...#./#..#./.##..",
    "K": "#...#/#..#./#.#../##.../#.#../#..#./#...#",
    "L": "#..../#..../#..../#..../#..../#..../#####",
    "M": "#...#/##.##/#.#.#/#.#.#/#...#/#...#/#...#",
    "N": "#...#/##..#/#.#.#/#..##/#...#/#...#/#...#",
    "O": ".###./#...#/#...#/#...#/#...#/#...#/.###.",
    "P": "####./#...#/#...#/####./#..../#..../#....",
    "Q": ".###./#...#/#...#/#...#/#.#.#/#..#./.##.#",
    "R": "####./#...#/#...#/####./#.#../#..#./#...#",
    "S": ".####/#..../#..../.###./....#/....#/####.",
    "T": "#####/..#../..#../..#../..#../..#../..#..",
    "U": "#...#/#...#/#...#/#...#/#...#/#...#/.###.",
    "V": "#...#/#...#/#...#/#...#/#...#/.#.#./..#..",
    "W": "#...#/#...#/#...#/#.#.#/#.#.#/##.##/#...#",
    "X": "#...#/#...#/.#.#./..#../.#.#./#...#/#...#",
    "Y": "#...#/#...#/.#.#./..#../..#../..#../..#..",
    "Z": "#####/....#/...#./..#../.#.../#..../#####",
    "0": ".###./#...#/#..##/#.#.#/##..#/#...#/.###.",
    "1": "..#../.##../..#../..#../..#../..#../.###.",
    "2": ".###./#...#/....#/...#./..#../.#.../#####",
    "3": "#####/...#./..#../...#./....#/#...#/.###.",
    "4": "...#./..##./.#.#./#..#./#####/...#./...#.",
    "5": "#####/#..../####./....#/....#/#...#/.###.",
    "6": "..##./.#.../#..../####./#...#/#...#/.###.",
    "7": "#####/....#/...#./..#../.#.../.#.../.#...",
    "8": ".###./#...#/#...#/.###./#...#/#...#/.###.",
    "9": ".###./#...#/#...#/.####/....#/...#./.##..",
    " ": "...../...../...../...../...../...../.....",
    ".": "...../...../...../...../...../.##../.##..",
    ",": "...../...../...../...../.##../.##../.#...",
    ":": "...../.##../.##../...../.##../.##../.....",
    "-": "...../...../...../#####/...../...../.....",
    "!": "..#../..#../..#../..#../..#../...../..#..",
    "?": ".###./#...#/....#/...#./..#../...../..#..",
    "'": "..#../..#../...../...../...../...../.....",
    "(": "...#./..#../.#.../.#.../.#.../..#../...#.",
    ")": ".#.../..#../...#./...#./...#./..#../.#...",
    "%": "##..#/##..#/...#./..#../.#.../#..##/#..##",
    "/": "....#/....#/...#./..#../.#.../#..../#....",
    "+": "...../..#../..#../#####/..#../..#../.....",
    "_": "...../...../...../...../...../...../#####",
    ">": ".#.../..#../...#./....#/...#./..#../.#...",
    "*": "...../#.#.#/.###./#####/.###./#.#.#/.....",
}

#: `glyph[char][row]` is a 5-bit mask, bit 4 = leftmost column.
FONT: dict[str, tuple[int, ...]] = {}
for _char, _art in _FONT_SRC.items():
    _rows = _art.split("/")
    if len(_rows) != GLYPH_H or any(len(r) != GLYPH_W for r in _rows):
        raise AssertionError(f"glyph {_char!r} is not {GLYPH_W}x{GLYPH_H}")
    FONT[_char] = tuple(
        sum(1 << (GLYPH_W - 1 - i) for i, px in enumerate(row) if px == "#") for row in _rows
    )

_FALLBACK = FONT["?"]


# --------------------------------------------------------------------------------------------
# Raster canvas
# --------------------------------------------------------------------------------------------

# Index into PALETTE below. Kept to 8 entries so the LZW minimum code size is 3.
BG, FG, DIM, OURS, THEIRS, WARN, BAR, EDGE = range(8)

PALETTE: tuple[tuple[int, int, int], ...] = (
    (0x12, 0x14, 0x1C),  # BG      page background
    (0xE6, 0xE8, 0xEF),  # FG      primary text
    (0x7C, 0x82, 0x96),  # DIM     secondary text
    (0x4E, 0xC9, 0xA0),  # OURS    our HP / our side
    (0xE0, 0x5C, 0x5C),  # THEIRS  their HP / their side
    (0xE8, 0xC8, 0x60),  # WARN    highlight (super effective, the win)
    (0x2A, 0x2E, 0x3C),  # BAR     empty bar / panel fill
    (0x39, 0x3F, 0x52),  # EDGE    rules and separators
)


class Canvas:
    """An 8-colour indexed raster. One byte per pixel, row-major -- GIF's own layout."""

    def __init__(self, width: int, height: int, fill: int = BG) -> None:
        self.width = width
        self.height = height
        self.px = bytearray([fill]) * (width * height)

    def rect(self, x: int, y: int, w: int, h: int, color: int) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes([color]) * (x1 - x0)
        for yy in range(y0, y1):
            start = yy * self.width + x0
            self.px[start : start + (x1 - x0)] = row

    def text(self, x: int, y: int, s: str, color: int, scale: int = 2) -> int:
        """Draw `s` uppercased at `(x, y)`; returns the x just past the last glyph."""
        cx = x
        for ch in s.upper():
            glyph = FONT.get(ch, _FALLBACK)
            for ry, bits in enumerate(glyph):
                if not bits:
                    continue
                for rx in range(GLYPH_W):
                    if bits & (1 << (GLYPH_W - 1 - rx)):
                        self.rect(cx + rx * scale, y + ry * scale, scale, scale, color)
            cx += (GLYPH_W + 1) * scale
        return cx


def text_width(s: str, scale: int = 2) -> int:
    return len(s) * (GLYPH_W + 1) * scale


# --------------------------------------------------------------------------------------------
# GIF89a writer
# --------------------------------------------------------------------------------------------


class _BitWriter:
    """LSB-first bit packer -- GIF packs LZW codes low bit first within each byte."""

    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0
        self._nbits = 0

    def write(self, code: int, width: int) -> None:
        self._acc |= code << self._nbits
        self._nbits += width
        while self._nbits >= 8:
            self._out.append(self._acc & 0xFF)
            self._acc >>= 8
            self._nbits -= 8

    def flush(self) -> bytes:
        if self._nbits > 0:
            self._out.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._out)


def lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    """GIF-flavoured LZW.

    The one subtlety is when the code width grows. The decoder is always one dictionary entry
    behind the encoder -- it can only finish an entry once it has seen the *next* code -- so the
    encoder must widen at `next_code == (1 << width) + 1` for the decoder, widening at
    `(1 << width)`, to stay in step. Off by one here and the file decodes to noise partway through
    rather than failing loudly, which is why `tests/test_make_battle_gif.py` carries an independent
    decoder and round-trips real frames through it.
    """
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    width = min_code_size + 1
    table: dict[bytes, int] = {bytes([i]): i for i in range(clear_code)}
    next_code = end_code + 1

    writer = _BitWriter()
    writer.write(clear_code, width)

    buf = b""
    for byte in indices:
        candidate = buf + bytes([byte])
        if candidate in table:
            buf = candidate
            continue
        writer.write(table[buf], width)
        if next_code < 4096:
            table[candidate] = next_code
            next_code += 1
            if next_code > (1 << width) and width < 12:
                width += 1
        else:
            writer.write(clear_code, width)
            table = {bytes([i]): i for i in range(clear_code)}
            next_code = end_code + 1
            width = min_code_size + 1
        buf = bytes([byte])

    if buf:
        writer.write(table[buf], width)
    writer.write(end_code, width)
    return writer.flush()


def _sub_blocks(data: bytes) -> bytes:
    """GIF data is carried in length-prefixed chunks of at most 255 bytes, zero-terminated."""
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i : i + 255]
        out.append(len(chunk))
        out += chunk
    out.append(0)
    return bytes(out)


def write_gif(
    path: Path,
    frames: list[Canvas],
    delays_cs: list[int],
    palette: tuple[tuple[int, int, int], ...] = PALETTE,
    loop: int = 0,
) -> int:
    """Write an animated GIF89a. Returns the byte length written."""
    if not frames:
        raise ValueError("no frames")
    if len(frames) != len(delays_cs):
        raise ValueError("one delay per frame")
    width, height = frames[0].width, frames[0].height

    # Palette size must be a power of two, >= 2, and the LZW minimum code size follows from it.
    bits = max(1, (len(palette) - 1).bit_length())
    table_size = 1 << bits
    min_code_size = max(2, bits)

    out = bytearray(b"GIF89a")
    out += bytes([width & 0xFF, width >> 8, height & 0xFF, height >> 8])
    out += bytes([0xF0 | (bits - 1), 0, 0])  # global table present, `bits`-bit colour
    for i in range(table_size):
        r, g, b = palette[i] if i < len(palette) else (0, 0, 0)
        out += bytes([r, g, b])

    # NETSCAPE2.0 application extension -- the only way to say "loop".
    out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + bytes([loop & 0xFF, loop >> 8]) + b"\x00"

    for canvas, delay in zip(frames, delays_cs, strict=True):
        if canvas.width != width or canvas.height != height:
            raise ValueError("all frames must share the canvas size")
        out += b"\x21\xf9\x04\x04" + bytes([delay & 0xFF, delay >> 8]) + b"\x00\x00"
        out += b"\x2c\x00\x00\x00\x00"
        out += bytes([width & 0xFF, width >> 8, height & 0xFF, height >> 8, 0x00])
        out += bytes([min_code_size])
        out += _sub_blocks(lzw_encode(bytes(canvas.px), min_code_size))

    out += b"\x3b"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return len(out)


# --------------------------------------------------------------------------------------------
# Frame layout
# --------------------------------------------------------------------------------------------

WIDTH, HEIGHT = 560, 260
SCALE = 2


def _hp_bar(canvas: Canvas, x: int, y: int, w: int, h: int, frac: float, color: int) -> None:
    canvas.rect(x - 1, y - 1, w + 2, h + 2, EDGE)
    canvas.rect(x, y, w, h, BAR)
    filled = int(round(w * max(0.0, min(1.0, frac))))
    if filled > 0:
        canvas.rect(x, y, filled, h, color)


def _ko_pips(canvas: Canvas, x: int, y: int, fainted: int, color: int) -> None:
    """Six pips per side; a filled pip is a Pokemon that has fainted."""
    for i in range(6):
        px = x + i * 14
        canvas.rect(px, y, 10, 10, color if i < fainted else BAR)


def render(frame: Frame, meta: Meta, us: str, first_turn: int, last_turn: int) -> Canvas:
    """Draw one frame. Our side is always the lower block, in green."""
    canvas = Canvas(WIDTH, HEIGHT)
    us_side = "p2" if meta.p2 == us else "p1"
    them_side = "p1" if us_side == "p2" else "p2"
    state = {"p1": frame.p1, "p2": frame.p2}
    them_name = meta.p1 if them_side == "p1" else meta.p2
    them_rating = meta.rating.get(them_side, "")

    canvas.text(16, 14, "ROTOMAI", OURS, SCALE)
    cursor = canvas.text(16 + text_width("ROTOMAI", SCALE) + 12, 14, "VS", DIM, SCALE)
    canvas.text(cursor + 12, 14, them_name, THEIRS, SCALE)

    tier = meta.tier or "GEN 9 OU"
    header = f"{tier}  {'RATED LADDER' if meta.rated else 'UNRATED'}"
    if them_rating:
        header += f"  OPPONENT ELO {them_rating}"
    canvas.text(16, 36, header, DIM, 1)
    canvas.rect(16, 52, WIDTH - 32, 1, EDGE)

    for idx, (side, color) in enumerate(((them_side, THEIRS), (us_side, OURS))):
        top = 68 + idx * 74
        who = "ROTOMAI" if side == us_side else them_name
        canvas.text(16, top, who, DIM, 1)
        canvas.text(16, top + 16, state[side].active or "-", FG, SCALE)
        _hp_bar(canvas, 250, top + 16, 220, 16, state[side].hp, color)
        canvas.text(480, top + 18, f"{round(state[side].hp * 100):d}%", color, 1)
        _ko_pips(canvas, 250, top + 42, state[side].fainted, color)
        canvas.text(250 + 6 * 14 + 8, top + 42, "FAINTED", DIM, 1)

    canvas.rect(16, 210, WIDTH - 32, 1, EDGE)
    canvas.text(16, 222, frame.caption, FG, SCALE)
    if frame.detail:
        color = WARN if frame.detail.endswith("!") or "WINS" in frame.caption else DIM
        canvas.text(16, 242, frame.detail, color, 1)
    span = f"TURN {frame.turn} / {last_turn}"
    canvas.text(WIDTH - 16 - text_width(span, 1), 242, span, DIM, 1)
    return canvas


# --------------------------------------------------------------------------------------------
# Replay ranking (the `--list` path OPERATIONS.md describes)
# --------------------------------------------------------------------------------------------


def score_replay(log: str, meta: Meta, us: str) -> tuple[int, int, int, bool]:
    """`(turns, our KOs, their KOs, we won)` -- enough to rank saved replays by watchability."""
    turns = 0
    faints = {"p1": 0, "p2": 0}
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        if parts[1] == "turn":
            turns = int(parts[2])
        elif parts[1] == "faint" and len(parts) >= 3:
            faints[_side_of(parts[2])] += 1
    us_side = "p2" if meta.p2 == us else "p1"
    them_side = "p1" if us_side == "p2" else "p2"
    return turns, faints[them_side], faints[us_side], meta.winner == us


def _list_replays(directory: Path, us: str) -> int:
    rows = []
    for path in sorted(directory.glob("*.html")):
        try:
            meta_and_log = extract_log(path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        meta = parse_meta(meta_and_log)
        turns, ours, theirs, won = score_replay(meta_and_log, meta, us)
        # A win in a handful of turns is an opponent forfeit, not a battle worth showing.
        if won and turns >= 10:
            rows.append((ours - theirs, turns, ours, theirs, path.name))
    rows.sort(reverse=True)
    if not rows:
        print(f"no won replays over 10 turns in {directory}", file=sys.stderr)
        return 1
    print(f"{'KOs':>7}  {'turns':>5}  file")
    for margin, turns, ours, theirs, name in rows[:20]:
        print(f"{ours:>3}-{theirs:<3}  {turns:>5}  {name}   (margin {margin:+d})")
    return 0


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def build_gif(
    replay: Path,
    out: Path,
    us: str = DEFAULT_US,
    turns: tuple[int, int] = DEFAULT_TURNS,
) -> tuple[int, int]:
    """Render `replay`'s `turns` window to `out`. Returns `(frame count, bytes written)`."""
    log = extract_log(replay.read_text(encoding="utf-8", errors="replace"))
    meta = parse_meta(log)
    if us not in (meta.p1, meta.p2):
        raise ValueError(f"{us!r} did not play this battle ({meta.p1} vs {meta.p2})")
    frames = build_frames(log, meta, us, turns[0], turns[1])
    if not frames:
        raise ValueError(f"no events in turns {turns[0]}-{turns[1]}")
    canvases = [render(f, meta, us, turns[0], turns[1]) for f in frames]
    size = write_gif(out, canvases, [f.delay_cs for f in frames])
    return len(frames), size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--us", default=DEFAULT_US, help="which player is the agent")
    parser.add_argument("--first-turn", type=int, default=DEFAULT_TURNS[0])
    parser.add_argument("--last-turn", type=int, default=DEFAULT_TURNS[1])
    parser.add_argument(
        "--list",
        type=Path,
        metavar="DIR",
        help="rank the saved replays in DIR by watchability and exit",
    )
    args = parser.parse_args(argv)

    if args.list is not None:
        return _list_replays(args.list, args.us)

    if not args.replay.exists():
        print(
            f"{args.replay} not found. `replays/` is gitignored, so a clone has none -- "
            "run the live client with `--save-replays` first.",
            file=sys.stderr,
        )
        return 1

    count, size = build_gif(args.replay, args.out, args.us, (args.first_turn, args.last_turn))
    verdict = "OK" if size <= SIZE_BUDGET_BYTES else "OVER BUDGET"
    print(f"{args.out}: {count} frames, {size / 1024:.1f} KiB ({verdict}, budget 3 MiB)")
    return 0 if size <= SIZE_BUDGET_BYTES else 1


if __name__ == "__main__":
    raise SystemExit(main())
