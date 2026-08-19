"""The write-up cannot drift from the record.

A paper is the one artifact in this repo that is not re-derived from a gate every time it is read,
so it is the one most able to go quietly stale -- a gate gets re-pinned, a rate moves in the fourth
decimal, and a number sits in prose for a year saying something the JSON no longer says.

These tests close that loop: every headline figure quoted in the paper and the blog post is read
back out of the committed result JSON and required to appear verbatim, and every figure is required
to still be what the current data produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
PAPER = ROOT / "paper" / "rotomai.md"
BLOG = ROOT / "blog" / "rotomai.md"

sys.path.insert(0, str(ROOT / "scripts"))

from make_figures import FIGURES  # noqa: E402


@pytest.fixture(scope="module")
def paper() -> str:
    return PAPER.read_text()


@pytest.fixture(scope="module")
def blog() -> str:
    return BLOG.read_text()


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# --------------------------------------------------------------------------- #
# The numbers
# --------------------------------------------------------------------------- #


def test_the_ladder_numbers_match_the_record(paper, blog):
    rating = _load("live_ladder_gen9ou_rating.json")
    merged = _load("live_ladder_gen9ou.json")
    elo, gxe = str(round(rating["elo"])), str(rating["gxe"])
    record = f"{rating['w']}–{rating['l']}"  # en dash, as the prose writes it
    for text in (paper, blog):
        assert f"Elo {elo}" in text
        assert f"GXE {gxe}%" in text
        assert record in text
    assert merged["verdict"] == "WEAK"
    assert "WEAK" in paper and "WEAK" in blog


def test_the_ladder_n_is_the_pre_registered_one(paper):
    prereg = _load("live_ladder_gen9ou_prereg.json")
    assert str(prereg["protocol"]["n_target"]) in paper
    bands = prereg["bands"]["WEAK"]
    # The band has to be quoted as declared, or "pre-registered WEAK" is unfalsifiable prose.
    assert bands["elo"].replace("< ", "< ") in paper.replace("Elo < 1200", "< 1200")


def test_the_simpleheuristics_headline_is_the_same_in_every_place_it_appears(paper, blog):
    """0.7303 is the THREE-run pooled figure and is in no single JSON -- one of the three runs was
    destroyed by the shared `--out` (docs/RESULTS.md, Build 28). So the check is consistency across
    the README, the paper and the blog, plus the interval that qualifies it."""
    readme = (ROOT / "README.md").read_text()
    for text in (readme, paper, blog):
        assert "0.7303" in text
    for text in (readme, paper):
        assert "0.7211" in text and "0.7394" in text
        assert "9,000" in text or "n=9000" in text


def test_the_surviving_runs_are_what_the_figure_plots():
    """Two of three reads survive as JSON; the figure must plot those and no invented third."""
    rates = []
    for name in ("seed_strength_gate_v26b_simpleheuristics.json",
                 "seed_strength_gate_v26b_simpleheuristics_paired.json"):
        build = next(iter(_load(name)["builds"].values()))
        rates.append(f"{build['rate']:.4f}")
    svg = (ASSETS / "fig_measurement.svg").read_text()
    for rate in rates:
        assert rate in svg, rate


def test_the_probe_seed_swing_is_backed_by_two_files_differing_only_in_seed(paper):
    """The paper reports a diagnostic retired for a 2.7x swing. Both probes must exist."""
    base = RESULTS / "grad_noise_diag_b22_stageB0_e10.json"
    other = RESULTS / "grad_noise_diag_b22_stageB0_e10_seed1.json"
    assert base.is_file() and other.is_file()
    assert "2.7" in paper and "2.2" in paper


def test_the_paper_does_not_repeat_the_uncounted_gate_figure(paper, blog):
    """"236 pre-registered gates" appears nowhere in this repository and must not be published."""
    assert "236" not in paper
    assert "236" not in blog


def test_the_counted_claims_are_the_ones_that_can_be_counted(paper):
    """The README already rotted this exact claim once (it said 240 against an actual 248).

    Counted from the INDEX, not the directory: the claim is about what is committed, and a gate
    output sitting un-added in a working tree is not part of the published record.
    """
    import re
    import subprocess

    try:
        listing = subprocess.run(
            ["git", "ls-files", "results/*.json"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git in a tarball
        pytest.skip("not a git checkout")
    committed = len([line for line in listing.splitlines() if line.strip()])
    if committed == 0:
        pytest.skip("no committed results (shallow or exported tree)")

    stated = re.search(r"(\d+) committed result JSONs", paper)
    assert stated, "the paper should state how many result files back it"
    assert int(stated.group(1)) == committed, (
        f"the paper says {stated.group(1)} committed result JSONs; the index holds {committed}"
    )


# --------------------------------------------------------------------------- #
# The figures
# --------------------------------------------------------------------------- #


def test_every_committed_figure_is_what_the_current_data_produces():
    """`make_figures.py --check` is the anti-drift gate; this runs it in-process."""
    from make_figures import main

    assert main(["--check"]) == 0, "run: python scripts/make_figures.py"


def test_the_paper_links_only_figures_that_exist(paper):
    import re

    linked = re.findall(r"!\[[^\]]*\]\((\.\./assets/[^)]+)\)", paper)
    assert linked, "the paper should carry its figures"
    for target in linked:
        assert (PAPER.parent / target).resolve().is_file(), target


def test_every_generated_figure_is_used(paper):
    for name in FIGURES:
        assert name in paper, f"{name} is generated but nothing references it"


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_a_figure_is_self_contained_and_themed(name):
    svg = (ASSETS / name).read_text()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<title>" in svg, "a figure needs an accessible name"
    # A strict-CSP or offline reader must not be asked to fetch anything. The SVG NAMESPACE is a
    # `http://www.w3.org/2000/svg` identifier and is not a fetch, so the check is on the attributes
    # that actually load something.
    for forbidden in ("xlink:href", "<image", "<script", 'href="http', 'src="http', "url(http"):
        assert forbidden not in svg, f"{name} references {forbidden}"
    # A transparent figure inherits the host page and becomes unreadable under one GitHub theme.
    assert 'fill="#12141c"' in svg


