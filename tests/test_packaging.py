"""The wheel must carry the data the code loads positionally.

Every file asserted here is opened by path relative to its module, not through
`importlib.resources` — `Path(__file__).with_name("data") / ...` in `features/vocab.py`,
`features/embed_prior.py`, `data/usage_prior.py`, `teambuilding/pool.py`, and three node drivers
spawned as subprocesses by `search/forward.py`, `search/fidelity.py`, `data/resim.py`. So a wheel
that omits them installs cleanly and then dies at the first vocab load, which no import-time check
catches.

Measured before `[tool.setuptools.package-data]` existed: the built wheel had 71 entries and
**none** of these eight. That is a package that cannot play a single battle.

Marked slow and skipped when `pip` cannot build offline, because it shells out to a real build.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

#: Loaded positionally at runtime. Filename only — the test asserts presence anywhere in the wheel,
#: because the package path is setuptools' business and the filename is ours.
RUNTIME_DATA = [
    "vocab_gen9.json",
    "id_priors_gen9.npz",
    "usage_gen9ou.json",
    "teams_gen9ou.packed",
    "teams_gen9vgc.packed",
    "forward_driver.js",
    "fidelity_driver.js",
    "resim_driver.js",
]


def test_every_runtime_data_file_is_declared_in_package_data():
    """The cheap half, always runs: the globs in pyproject must cover every file's extension.

    Not a substitute for building — a glob can be present and still not match — but it fails fast
    and without a network, which the build test cannot promise.
    """
    text = (_ROOT / "pyproject.toml").read_text()
    assert "[tool.setuptools.package-data]" in text, "no package-data section at all"
    for suffix in (".json", ".npz", ".packed", ".js"):
        assert f"*{suffix}" in text, f"no package-data glob covers {suffix}"


@pytest.mark.slow
def test_a_built_wheel_actually_contains_them(tmp_path):
    """The real check: export the tracked tree, build a wheel, look inside it."""
    export = tmp_path / "src"
    export.mkdir()
    tar = subprocess.run(["git", "archive", "HEAD"], cwd=_ROOT, capture_output=True)
    if tar.returncode != 0:
        pytest.skip("not a git checkout")
    subprocess.run(["tar", "-x", "-C", str(export)], input=tar.stdout, check=True)
    # Use the working-tree pyproject, so an uncommitted packaging change is what gets tested.
    (export / "pyproject.toml").write_bytes((_ROOT / "pyproject.toml").read_bytes())

    build = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-w", str(tmp_path / "dist"), "."],
        cwd=export, capture_output=True, text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable: {build.stderr.strip().splitlines()[-1:]}")

    wheels = list((tmp_path / "dist").glob("*.whl"))
    assert wheels, "build reported success but produced no wheel"
    names = zipfile.ZipFile(wheels[0]).namelist()
    missing = [d for d in RUNTIME_DATA if not any(d in n for n in names)]
    assert not missing, (
        f"{len(missing)} runtime data file(s) absent from the wheel: {missing}. "
        f"`pip install lategame` would fail at first use. Check "
        f"[tool.setuptools.package-data] in pyproject.toml."
    )
