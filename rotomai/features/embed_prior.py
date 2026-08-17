"""Dex-feature identity priors for the R-ENCODE embeddings (pre-initialization).

The data-scaling sweep (docs/RESULTS.md) showed identity embeddings are *starved*: they
emerge with scale but, at the exhausted replay ceiling, land just short of the no-embedding
baseline. The cheap remedy that needs no more scrape is to **warm-start** the species/move
embedding tables from what each entity actually *is* -- base stats + types for species,
power/accuracy/type/category for moves -- so the policy doesn't have to rediscover identity
semantics from a handful of winners-only turns.

Mirrors ``features.vocab``: built once from poke-env ``GenData`` (maintenance path), frozen
into a committed ``data/id_priors_gen9.npz`` stamped with ``vocab_version`` (drift guard),
and read back as plain numpy at train time -- no ``GenData``/node dependency on the hot path.

The committed artifact stores the **raw standardized feature matrices** (decoupled from the
embedding width); ``load_id_priors`` projects them to ``id_embed_dim`` with a fixed-seed
Gaussian and rescales to the default ``nn.Embedding`` std, so the prior arm starts at the
*same scale* as the random arm -- only the structure differs. Only ``species`` and ``moves``
are seeded; items/abilities lack structured dex features and stay random.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from poke_env.battle import MoveCategory, PokemonType

from rotomai.features.vocab import load_vocab, vocab_version

# Only the two fields with rich structured dex features are seeded; items/abilities
# (no usable GenData features) keep random init.
PRIOR_FIELDS = ("species", "moves")
_PRIORS_PATH = Path(__file__).with_name("data") / "id_priors_gen9.npz"

# Fixed projection seeds (one per field) -> deterministic, reproducible warm-start.
_PROJ_SEED = {"species": 1_000_003, "moves": 1_000_033}

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_TYPES = list(PokemonType)
_TYPE_INDEX = {t: i for i, t in enumerate(_TYPES)}
_CATEGORIES = list(MoveCategory)
_CATEGORY_INDEX = {c: i for i, c in enumerate(_CATEGORIES)}
_MOVE_NUMERIC = 4  # base_power, accuracy, priority, pp


def _type_index(name: str | None) -> int | None:
    """Map a GenData type string (e.g. ``"Dragon"``) to its enum slot, or None (glitch types)."""
    if not name:
        return None
    try:
        return _TYPE_INDEX[PokemonType[name.upper()]]
    except KeyError:
        return None


def _standardize(mat: np.ndarray, cols: list[int]) -> None:
    """Z-score the given numeric columns over real rows (index >= 1) in place."""
    real = mat[1:]
    for j in cols:
        col = real[:, j]
        std = col.std()
        mat[1:, j] = (col - col.mean()) / std if std > 1e-8 else col - col.mean()


def _species_features(pokedex: dict, table: dict[str, int]) -> np.ndarray:
    """``(vocab_size, 6 stats + |types|)`` matrix; row 0 (UNK / padding_idx) stays zero."""
    feat_dim = len(_STAT_KEYS) + len(_TYPES)
    mat = np.zeros((len(table) + 1, feat_dim), dtype=np.float64)
    for id_str, idx in table.items():
        entry = pokedex[id_str]
        stats = entry.get("baseStats", {})
        for j, key in enumerate(_STAT_KEYS):
            mat[idx, j] = float(stats.get(key, 0))
        for t in entry.get("types", []):
            ti = _type_index(t)
            if ti is not None:
                mat[idx, len(_STAT_KEYS) + ti] = 1.0
    _standardize(mat, list(range(len(_STAT_KEYS))))
    mat[0] = 0.0
    return mat.astype(np.float32)


def _move_features(moves: dict, table: dict[str, int]) -> np.ndarray:
    """``(vocab_size, 4 numeric + |cat| + |types|)`` matrix; row 0 (UNK) stays zero."""
    feat_dim = _MOVE_NUMERIC + len(_CATEGORIES) + len(_TYPES)
    mat = np.zeros((len(table) + 1, feat_dim), dtype=np.float64)
    for id_str, idx in table.items():
        mv = moves[id_str]
        acc = mv.get("accuracy")
        mat[idx, 0] = float(mv.get("basePower") or 0)
        mat[idx, 1] = 1.0 if acc is True else (float(acc) / 100.0 if acc else 0.0)
        mat[idx, 2] = float(mv.get("priority") or 0)
        mat[idx, 3] = float(mv.get("pp") or 0)
        cat = mv.get("category")
        if cat:
            mat[idx, _MOVE_NUMERIC + _CATEGORY_INDEX[MoveCategory[cat.upper()]]] = 1.0
        ti = _type_index(mv.get("type"))
        if ti is not None:
            mat[idx, _MOVE_NUMERIC + len(_CATEGORIES) + ti] = 1.0
    _standardize(mat, [0, 2, 3])  # accuracy already in [0, 1]
    mat[0] = 0.0
    return mat.astype(np.float32)


# --------------------------------------------------------------------------- #
# Build / regenerate (maintenance tool -- needs GenData; not on any train path).
# --------------------------------------------------------------------------- #
def build_id_priors() -> dict[str, np.ndarray]:
    """Raw standardized dex-feature matrices keyed by field, aligned to vocab indices."""
    from poke_env.data import GenData

    gd = GenData.from_gen(9)
    vocab = load_vocab()
    return {
        "species": _species_features(gd.pokedex, vocab.tables["species"]),
        "moves": _move_features(gd.moves, vocab.tables["moves"]),
    }


def write_id_priors(path: Path = _PRIORS_PATH) -> str:
    """Build and freeze the priors npz (stamped with ``vocab_version``); returns the stamp."""
    priors = build_id_priors()
    version = vocab_version()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vocab_version=np.asarray(version),
        species=priors["species"],
        moves=priors["moves"],
    )
    load_id_priors.cache_clear()
    return version


# --------------------------------------------------------------------------- #
# Load (train path -- numpy only; projects raw features to the embedding width).
# --------------------------------------------------------------------------- #
def _project(feats: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Random-project ``(V, feat_dim)`` -> ``(V, dim)`` and rescale to nn.Embedding std (~1)."""
    rng = np.random.default_rng(seed)
    emb = feats.astype(np.float64) @ rng.standard_normal((feats.shape[1], dim))
    std = emb[1:].std()
    if std > 1e-8:
        emb /= std
    emb[0] = 0.0  # keep the UNK / padding_idx row at zero
    return emb.astype(np.float32)


@lru_cache(maxsize=4)
def load_id_priors(id_embed_dim: int) -> dict[str, np.ndarray]:
    """Per-field ``(vocab_size, id_embed_dim)`` warm-start matrices for ``species``/``moves``."""
    data = np.load(_PRIORS_PATH)
    stored = str(data["vocab_version"])
    if stored != vocab_version():
        raise ValueError(
            f"id_priors vocab_version {stored} != live {vocab_version()} "
            "-- rebuild via embed_prior.write_id_priors()"
        )
    return {cat: _project(data[cat], id_embed_dim, _PROJ_SEED[cat]) for cat in PRIOR_FIELDS}
