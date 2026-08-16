"""The release manifest must be derived, and must stay in step with the claims it backs.

A hand-kept list of "weights worth shipping" is the same object as a hand-kept count of mypy errors
or results files: correct once, then quietly wrong. This one is generated from
`check_artifacts.HEADLINE`, and these tests pin that it cannot drift away from it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _ROOT / "results" / "release_assets.json"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "release_assets", _ROOT / "scripts" / "release_assets.py"
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_runnable_asset_is_cited_by_a_headline():
    """A weight worth shipping is a weight some published claim stands on. Anything in RUNNABLE
    that no headline cites is either an unpublished claim or a stale path."""
    ra = _mod()
    cited = set(ra._cited_by_headlines())
    for path in ra.RUNNABLE:
        assert path in cited, f"{path} is marked runnable but backs no headline"


def test_the_manifest_covers_one_runnable_weight_per_format():
    """The point of shipping weights is that someone can PLAY each format. A release that carried
    only gen9ou weights would leave the doubles half of G4 unrunnable."""
    ra = _mod()
    blurbs = " ".join(ra.RUNNABLE.values())
    assert "gen9ou" in blurbs
    assert "gen9vgc2025regi" in blurbs


@pytest.mark.skipif(not _MANIFEST.exists(), reason="manifest not generated")
def test_the_committed_manifest_lists_exactly_what_the_headlines_cite():
    """The drift check, and it must run WHERE THE WEIGHTS ARE NOT -- which is CI, and a clone.

    The first version of this compared the committed file to `build_manifest()`, which skips
    checkpoints absent from disk. That passes only on the machine holding the weights and fails on
    every bare clone, so it tested the environment rather than the manifest. `_cited_by_headlines`
    reads committed `results/` alone and needs no checkpoint present, so the comparison is the one
    actually worth making: does the shipped list still match the claims it is derived from?
    """
    ra = _mod()
    committed = json.loads(_MANIFEST.read_text())
    assert sorted(a["path"] for a in committed["assets"]) == sorted(ra._cited_by_headlines())
    for a in committed["assets"]:
        assert a["backs"] == ra._cited_by_headlines()[a["path"]]


@pytest.mark.skipif(not _MANIFEST.exists(), reason="manifest not generated")
def test_every_asset_carries_a_digest_and_a_size():
    """Without these the manifest cannot answer the question it exists for: is the file I just
    downloaded the file this repository published? A checkpoint is a pickle and `torch.load` runs
    code from it, so "trust the release page" is not a posture."""
    m = json.loads(_MANIFEST.read_text())
    assert m["assets"], "an empty manifest would pass every other test here"
    for a in m["assets"]:
        assert len(a["sha256"]) == 64 and int(a["sha256"], 16) >= 0
        assert a["bytes"] > 0
