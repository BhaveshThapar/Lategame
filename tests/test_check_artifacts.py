"""A published headline must fail loudly; a superseded intermediate must not.

The README states, in counted form, that no headline number depends on a checkpoint that scratch
teardown ate. That claim is only worth writing down if something re-derives it -- and only useful if
its exit status distinguishes the two cases, because 58 of the 121 cited paths are *already* gone
and a check that reddened on those would be turned off within a week.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_artifacts  # type: ignore[import-not-found]
from check_artifacts import HEADLINE, audit  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent


def _tree(tmp_path: Path, gates: dict[str, list[str]], on_disk: tuple[str, ...]) -> None:
    """`gates` maps a results filename to the paths it cites; `on_disk` is what actually exists."""
    (tmp_path / "results").mkdir(exist_ok=True)
    for fname, refs in gates.items():
        (tmp_path / "results" / fname).write_text(
            json.dumps({"records": [{"best_checkpoint": r} for r in refs]})
        )
    for rel in on_disk:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 8)


@pytest.fixture
def one_headline(monkeypatch):
    """Pin HEADLINE to a single synthetic claim so the fixtures do not depend on the real tree."""
    monkeypatch.setattr(check_artifacts, "HEADLINE", {"synthetic claim": ("head.json",)})


def test_a_present_headline_passes(tmp_path, one_headline):
    _tree(
        tmp_path,
        {"head.json": ["checkpoints/arm_s0/iter_09.pt"]},
        ("checkpoints/arm_s0/iter_09.pt",),
    )
    report = audit(tmp_path / "results", tmp_path / "checkpoints")

    assert report["headlines"]["synthetic claim"]["ok"]
    assert report["missing"] == []


def test_a_missing_headline_checkpoint_fails(tmp_path, one_headline):
    """The case the script exists for: a published number can no longer be re-read here."""
    _tree(tmp_path, {"head.json": ["checkpoints/arm_s0/iter_09.pt"]}, ())
    report = audit(tmp_path / "results", tmp_path / "checkpoints")

    h = report["headlines"]["synthetic claim"]
    assert not h["ok"]
    assert h["missing_checkpoints"] == ["checkpoints/arm_s0/iter_09.pt"]


def test_a_missing_headline_RECORD_fails_too(tmp_path, one_headline):
    """A renamed or deleted gate file is the same failure as a deleted checkpoint -- the claim is
    unbacked either way, and an empty citation set would otherwise read as vacuously OK."""
    _tree(tmp_path, {"other.json": ["checkpoints/arm_s0/iter_09.pt"]}, ())
    report = audit(tmp_path / "results", tmp_path / "checkpoints")

    h = report["headlines"]["synthetic claim"]
    assert not h["ok"]
    assert h["missing_records"] == ["head.json"]


def test_a_missing_NON_headline_checkpoint_is_reported_but_not_a_failure(tmp_path, one_headline):
    """58 of the real tree's 121 cited paths are already gone. They back superseded builds whose
    numbers are still in the JSON, so they are a documented state -- reported, never red."""
    _tree(
        tmp_path,
        {
            "head.json": ["checkpoints/arm_s0/iter_09.pt"],
            "old.json": ["checkpoints/dead_s0/iter_04.pt"],
        },
        ("checkpoints/arm_s0/iter_09.pt",),
    )
    report = audit(tmp_path / "results", tmp_path / "checkpoints")

    assert report["headlines"]["synthetic claim"]["ok"]
    assert report["missing"] == ["checkpoints/dead_s0/iter_04.pt"]
    assert report["records_citing_missing"] == ["old.json"]


def test_the_checkpoints_root_is_honoured_not_assumed(tmp_path, one_headline):
    """Cited paths are written `checkpoints/...` relative to the repo root, so pointing the check
    at a staged tree must rewrite that first component rather than report a wall of misses."""
    _tree(tmp_path, {"head.json": ["checkpoints/arm_s0/iter_09.pt"]}, ())
    staged = tmp_path / "staged"
    (staged / "arm_s0").mkdir(parents=True)
    (staged / "arm_s0" / "iter_09.pt").write_bytes(b"x" * 8)

    assert audit(tmp_path / "results", staged)["headlines"]["synthetic claim"]["ok"]


def test_every_headline_maps_to_a_gate_file_that_cites_something():
    """The half of the check that survives a bare clone, and so the half CI can run.

    `results/` is committed and `checkpoints/` is not, so a runner cannot ask whether the weights
    exist -- but it can ask whether each headline still names a gate file that is present and cites
    at least one checkpoint. That catches the quiet failure: a renamed or deleted record empties the
    citation set, and an empty set would otherwise satisfy every "nothing missing" assertion.
    """
    report = audit(ROOT / "results", ROOT / "checkpoints")

    for claim, h in report["headlines"].items():
        assert not h["missing_records"], f"{claim}: gate file(s) gone: {h['missing_records']}"
        assert h["checkpoints"], f"{claim}: its records cite no checkpoint at all"


@pytest.mark.skipif(not (ROOT / "checkpoints").is_dir(), reason="no local checkpoint tree")
def test_every_published_headline_resolves_against_the_real_tree():
    """The integration read, and the one the README's Artifacts section is only true because of."""
    report = audit(ROOT / "results", ROOT / "checkpoints")

    assert report["referenced"], "no checkpoints cited at all -- the scan is broken, not the tree"
    unbacked = {claim: h for claim, h in report["headlines"].items() if not h["ok"]}
    assert not unbacked, f"headline(s) no longer reproducible: {unbacked}"
    assert set(report["headlines"]) == set(HEADLINE)
