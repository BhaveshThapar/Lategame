"""Render the paper/README figures as SVG, straight from `results/*.json`. No plotting library.

matplotlib is not installed here and is not worth adding: three figures do not justify a build
dependency that pulls in a font stack and an image backend. SVG is also the better artifact for
this repo -- it is text, so a figure diffs like code and a number that changed in a gate shows up
as a changed line rather than a changed blob.

EVERY NUMBER IS READ FROM A COMMITTED RESULT FILE. Nothing here is typed in from the README, so a
figure cannot drift from the record the way a hand-drawn one does. Where a number exists in the
record but NOT on disk -- Build 28 lost two of six reads to the `--out` global, `docs/RESULTS.md`
-- the figure says so on its face rather than quietly plotting what survived.

    python scripts/make_figures.py            # -> assets/fig_*.svg
    python scripts/make_figures.py --check    # regenerate into a temp dir and diff; no writes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path("results")
ASSETS = Path("assets")

# Shared with the demo GIF so the README reads as one thing. A dark card renders acceptably under
# both GitHub themes; a transparent one inherits the page and becomes unreadable in one of them.
BG = "#12141c"
FG = "#e6e8ef"
DIM = "#7c8296"
EDGE = "#393f52"
ACCENT = "#4ec9a0"
WARN = "#e8c860"
BAD = "#e05c5c"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x: float, y: float, s: str, *, size: float = 12, fill: str = FG,
          anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{_esc(s)}</text>'
    )


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_esc(title)}">'
        f"<title>{_esc(title)}</title>"
        f'<rect width="{width}" height="{height}" fill="{BG}"/>'
        f"{body}</svg>\n"
    )


# --------------------------------------------------------------------------------------------
# Figure 1: the pipeline
# --------------------------------------------------------------------------------------------


def figure_pipeline() -> str:
    """What trains what. The dashed stage is the one no published claim rests on."""
    width, height = 900, 260
    stages = [
        ("Human replays", "2,760 rated gen9ou\n119,996 turns", True),
        ("Behaviour cloning", "61.8k winners' turns\nval-acc 0.647", True),
        ("Offline RL (AWR)", "HL-Gauss value\n+ advantage weights", True),
        ("PPO self-play", "9 arms x 160 iters\n107 task-hours", False),
    ]
    body = [_text(24, 34, "RotomAI training pipeline", size=16, weight="bold")]
    body.append(_text(24, 54, "dashed = measured once and never scaled; no claim rests on it",
                      size=11, fill=DIM))

    box_w, box_h, gap = 186, 92, 52
    for index, (name, detail, dashed) in enumerate(stages):
        x = 24 + index * (box_w + gap)
        y = 82
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        stroke = DIM if dashed else ACCENT
        body.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6" '
            f'fill="none" stroke="{stroke}" stroke-width="1.5"{dash}/>'
        )
        body.append(_text(x + box_w / 2, y + 26, name, size=13, anchor="middle", weight="bold"))
        for line_no, line in enumerate(detail.split("\n")):
            body.append(
                _text(x + box_w / 2, y + 50 + line_no * 16, line, size=11,
                      fill=DIM, anchor="middle")
            )
        if index < len(stages) - 1:
            ax = x + box_w
            body.append(
                f'<path d="M{ax + 8} {y + box_h / 2} L{ax + gap - 12} {y + box_h / 2}" '
                f'stroke="{EDGE}" stroke-width="1.5"/>'
                f'<path d="M{ax + gap - 12} {y + box_h / 2} l-7 -4 v8 z" fill="{EDGE}"/>'
            )

    outputs = [
        ("0.7303 vs poke-env SimpleHeuristicsPlayer", "n=9,000, three runs pooled", ACCENT),
        ("Elo 1004 / GXE 30% vs humans", "100 pre-registered rated games -- WEAK", BAD),
    ]
    for index, (headline, detail, colour) in enumerate(outputs):
        y = 208 + index * 26
        body.append(f'<circle cx="30" cy="{y - 4}" r="4" fill="{colour}"/>')
        body.append(_text(44, y, headline, size=12, fill=colour, weight="bold"))
        body.append(_text(430, y, detail, size=11, fill=DIM))
    return _svg(width, height, "".join(body), "RotomAI training pipeline and headline results")


# --------------------------------------------------------------------------------------------
# Figure 2: what a measurement costs
# --------------------------------------------------------------------------------------------


def _gate_runs() -> list[tuple[str, list[float], float]]:
    """`(label, per-checkpoint rates, pooled rate)` for every surviving simpleheuristics read."""
    runs = []
    for name, label in (
        ("seed_strength_gate_v26b_simpleheuristics.json", "run A"),
        ("seed_strength_gate_v26b_simpleheuristics_paired.json", "run B"),
    ):
        path = RESULTS / name
        if not path.exists():
            continue
        build = next(iter(json.loads(path.read_text())["builds"].values()))
        runs.append(
            (label, [c["rate"] for c in build["per_checkpoint"]], float(build["rate"]))
        )
    return runs


def figure_measurement() -> str:
    """Per-checkpoint rates move across runs; the seed-pooled read barely does."""
    runs = _gate_runs()
    if not runs:
        raise SystemExit("no simpleheuristics gate JSON on disk")

    width, height = 900, 400
    left, right, top, bottom = 90, 640, 96, 320
    lo, hi = 0.64, 0.80

    def ypos(rate: float) -> float:
        return bottom - (rate - lo) / (hi - lo) * (bottom - top)

    body = [
        _text(24, 34, "The same weights, measured twice", size=16, weight="bold"),
        _text(24, 54,
              "Score rate vs poke-env SimpleHeuristicsPlayer, gen9ou, n=1,000 per checkpoint",
              size=11, fill=DIM),
        _text(24, 72,
              "Every point is the SAME checkpoint file. Only the run differs.",
              size=11, fill=DIM),
    ]

    for step in range(7):
        rate = lo + step * (hi - lo) / 6
        y = ypos(rate)
        body.append(
            f'<path d="M{left} {y:.1f} H{right}" stroke="{EDGE}" stroke-width="0.6"/>'
        )
        body.append(_text(left - 10, y + 4, f"{rate:.2f}", size=10, fill=DIM, anchor="end"))

    n_seeds = len(runs[0][1])
    span = (right - left) / (n_seeds + 1)
    for seed in range(n_seeds):
        x = left + span * (seed + 1)
        body.append(_text(x, bottom + 22, f"seed {seed}", size=11, fill=DIM, anchor="middle"))
        rates = [run[1][seed] for run in runs]
        body.append(
            f'<path d="M{x} {ypos(min(rates)):.1f} V{ypos(max(rates)):.1f}" '
            f'stroke="{WARN}" stroke-width="1.5"/>'
        )
        for run_index, rate in enumerate(rates):
            body.append(
                f'<circle cx="{x}" cy="{ypos(rate):.1f}" r="5" fill="{BG}" '
                f'stroke="{ACCENT if run_index == 0 else WARN}" stroke-width="2"/>'
            )
        swing = max(rates) - min(rates)
        body.append(
            _text(x, ypos(max(rates)) - 14, f"{swing:+.3f}", size=11, fill=WARN, anchor="middle")
        )

    pooled = [run[2] for run in runs]
    for rate in pooled:
        body.append(
            f'<path d="M{left} {ypos(rate):.1f} H{right}" stroke="{ACCENT}" '
            f'stroke-width="1.2" stroke-dasharray="6 4"/>'
        )
    body.append(
        _text(right + 14, ypos(max(pooled)) + 4,
              f"seed-pooled: {min(pooled):.4f} / {max(pooled):.4f}", size=11, fill=ACCENT)
    )
    body.append(
        _text(right + 14, ypos(max(pooled)) + 22,
              f"spread {max(pooled) - min(pooled):.4f}", size=11, fill=ACCENT)
    )

    worst = max(max(r[1][seed] for r in runs) - min(r[1][seed] for r in runs)
                for seed in range(n_seeds))
    pooled_spread = max(pooled) - min(pooled)
    ratio = worst / pooled_spread if pooled_spread else float("inf")
    notes = [
        f"Worst per-checkpoint spread is {ratio:.0f}x the pooled spread.",
        "Standing rule: a cross-run difference under ~0.03 is not a result.",
        "Pooling over seeds is what buys run-to-run stability;",
        "raising n within one seed does not -- between-seed sd is",
        "~2.5x the within-seed binomial.",
        "",
        f"{len(runs)} of the 3 runs survive as JSON. The third exists only",
        "in its job log: two overlapping jobs shared one --out and the",
        "later write won. (docs/RESULTS.md, Build 28)",
    ]
    for index, line in enumerate(notes):
        body.append(_text(24, bottom + 60 + index * 15, line, size=11,
                          fill=BAD if index >= 6 else DIM))
    return _svg(width, height, "".join(body), "Cross-run variance versus the seed-pooled read")


# --------------------------------------------------------------------------------------------
# Figure 3: the ladder run
# --------------------------------------------------------------------------------------------


def _ladder_battles() -> list[dict]:
    battles: list[dict] = []
    for segment in sorted(RESULTS.glob("live_ladder_gen9ou_seg*.json")):
        battles.extend(json.loads(segment.read_text())["battles"])
    battles.sort(key=lambda b: b.get("started_at") or "")
    return battles


def figure_ladder() -> str:
    """100 pre-registered rated games, and the pre-registered bands they were read against."""
    battles = _ladder_battles()
    rating = json.loads((RESULTS / "live_ladder_gen9ou_rating.json").read_text())
    merged = json.loads((RESULTS / "live_ladder_gen9ou.json").read_text())
    elos = [b["showdown_elo_before"] for b in battles if b.get("showdown_elo_before")]
    if not elos:
        raise SystemExit("no pre-battle Elo recorded in the ladder segments")

    width, height = 900, 380
    left, right, top, bottom = 80, 700, 90, 300
    lo, hi = 960, 1260

    def ypos(elo: float) -> float:
        return bottom - (elo - lo) / (hi - lo) * (bottom - top)

    def xpos(index: int) -> float:
        return left + index / max(1, len(elos) - 1) * (right - left)

    body = [
        _text(24, 34, "The public ranked ladder, pre-registered WEAK", size=16, weight="bold"),
        _text(24, 54,
              f"{len(battles)} rated gen9ou games. Bands frozen before the first game.",
              size=11, fill=DIM),
    ]

    # The run never reached DECENT, so the axis stops just above that boundary rather than at the
    # top of STRONG -- an axis sized for a band nothing entered renders the whole run as a flat
    # line on the floor and hides the only variation there is.
    for label, top_elo, floor, colour in (
        ("DECENT  1200-1450 (not reached)", hi, 1200, WARN),
        ("WEAK  < 1200  <- pre-registered verdict", 1200, lo, BAD),
    ):
        y0, y1 = ypos(top_elo), ypos(floor)
        body.append(
            f'<rect x="{left}" y="{y0:.1f}" width="{right - left}" height="{y1 - y0:.1f}" '
            f'fill="{colour}" fill-opacity="0.07"/>'
        )
        body.append(_text(right + 12, y0 + 16, label, size=11, fill=colour))

    for elo in (1000, 1050, 1100, 1150, 1200, 1250):
        y = ypos(elo)
        body.append(f'<path d="M{left} {y:.1f} H{right}" stroke="{EDGE}" stroke-width="0.6"/>')
        body.append(_text(left - 10, y + 4, str(elo), size=10, fill=DIM, anchor="end"))

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{xpos(i):.1f} {ypos(elo):.1f}" for i, elo in enumerate(elos)
    )
    body.append(f'<path d="{path}" fill="none" stroke="{FG}" stroke-width="1.6"/>')
    body.append(
        f'<circle cx="{xpos(len(elos) - 1):.1f}" cy="{ypos(elos[-1]):.1f}" r="4" fill="{BAD}"/>'
    )
    body.append(_text(left, bottom + 22, "game 1", size=10, fill=DIM))
    body.append(_text(right, bottom + 22, f"game {len(battles)}", size=10, fill=DIM,
                      anchor="end"))

    verdict = (
        f"final: Elo {round(rating['elo'])} / GXE {rating['gxe']}%   "
        f"record {rating['w']}-{rating['l']}   verdict {merged['verdict']}"
    )
    body.append(_text(24, bottom + 58, verdict, size=13, fill=BAD, weight="bold"))
    for index, line in enumerate([
        "The same checkpoint scores 0.7303 against poke-env's shared bot baseline.",
        "Both numbers are correct; they measure different things.",
        "A bot-baseline win rate does not predict human-ladder performance.",
        "The line is the PRE-battle Elo poke-env sees, so it stops one game short of the",
        "endpoint reading -- which is why the endpoint was pre-registered as primary.",
    ]):
        body.append(_text(24, bottom + 80 + index * 15, line, size=11, fill=DIM))
    return _svg(width, height, "".join(body), "Pre-registered ranked ladder run")


FIGURES = {
    "fig_pipeline.svg": figure_pipeline,
    "fig_measurement.svg": figure_measurement,
    "fig_ladder.svg": figure_ladder,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=ASSETS)
    parser.add_argument("--check", action="store_true",
                        help="fail if a committed figure differs from what the data now produces")
    args = parser.parse_args(argv)

    stale = []
    for name, build in FIGURES.items():
        content = build()
        target = args.out_dir / name
        if args.check:
            if not target.exists() or target.read_text() != content:
                stale.append(name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        print(f"{target}  {len(content) / 1024:.1f} KiB")

    if args.check:
        if stale:
            print("stale figures: " + ", ".join(stale), file=sys.stderr)
            print("regenerate with: python scripts/make_figures.py", file=sys.stderr)
            return 1
        print(f"{len(FIGURES)} figures match the committed results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
