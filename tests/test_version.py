"""The version lives in three files and they must agree.

`pyproject.toml`, `rotomai/__init__.py` and `CITATION.cff` each carry it, so a release bump that
touches one leaves the others quietly wrong -- the same drift that let a "2 known mypy errors"
footnote and a "221 results files" count survive this repo for months. This is the cheap version of
the fix: a bar the machine runs, so the three cannot disagree.
"""

import re
import tomllib
from pathlib import Path

import rotomai

_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["version"]


def _citation_version() -> str:
    text = (_ROOT / "CITATION.cff").read_text()
    m = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', text, re.M)
    assert m, "CITATION.cff has no version field"
    return m.group(1).strip()


def test_the_three_version_strings_agree():
    assert rotomai.__version__ == _pyproject_version() == _citation_version()


def test_the_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _pyproject_version())
