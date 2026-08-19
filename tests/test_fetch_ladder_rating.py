"""The endpoint reader: id normalisation and the payload shapes, with no network.

The fetch itself is one `urlopen` and is not worth mocking a socket for; what IS worth pinning is
`to_id`, because the ratings endpoint keys on it and a wrong id returns a 200 with an empty
`ratings` object rather than an error -- which reads as "no games yet" instead of "wrong user".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_ladder_rating import to_id  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("RotomLover12", "rotomlover12"),
        ("Rotom Lover 12", "rotomlover12"),
        ("Rotom-Lover_12", "rotomlover12"),
        ("ROTOMLOVER12", "rotomlover12"),
        ("rotomlover12", "rotomlover12"),
    ],
)
def test_to_id_matches_showdowns_normalisation(raw, expected):
    assert to_id(raw) == expected


def test_names_that_differ_only_in_punctuation_are_the_same_account():
    """Showdown treats these as one user, so the reader must too -- otherwise a run under
    'Rotom Lover 12' would look up an account with no ladder history and report a pilot."""
    assert to_id("Rotom Lover 12") == to_id("RotomLover12")
