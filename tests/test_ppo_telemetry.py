"""The trust-region certificate must be recoverable from a run log.

Build 21 could not audit Build 20's early iterations because that telemetry was never
persisted -- only the win-rate curve was. These tests pin the parse so the certificate a
build's verdict leans on ("the trust region never bound, so a NULL cannot be blamed on it")
survives the run that produced it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ppo_telemetry import parse_log, summarize  # type: ignore[import-not-found]


def _stats_line(it: int, approx_kl: float, vmae: float, epochs: int) -> str:
    """Byte-for-byte the line ppo.py:256-262 prints -- if that f-string drifts, this breaks."""
    return (
        f"  ppo iter {it:>3}  turns 5983  "
        f"pi_loss -0.0181  v_loss 2.2181  "
        f"entropy 1.131  approx_kl {approx_kl:.4f}  "
        f"clip 0.245  R[-6.86,7.77]  "
        f"vmae {vmae:.3f}  epochs {epochs}  "
        f"ent_coef 0.0084  lr 2.17e-04"
    )


# A real run's opening: iter 1 overshoots the KL bar and early-stops after 1 of 4 epochs.
LOG = "\n".join(
    [
        "init checkpoints/offrl_gen9ou_wide_s0.pt | seeds [0] | iters 50",
        "training on mps",
        _stats_line(1, 0.0517, 0.822, 1),
        "iter   1  vs_random 0.500  vs_simpleheuristics 0.100  vs_heuristic 0.030",
        _stats_line(2, 0.0225, 0.800, 4),
        _stats_line(3, 0.0259, 0.680, 4),
        "",
    ]
)


def _write(tmp_path, text=LOG):
    p = tmp_path / "run.log"
    p.write_text(text)
    return p


def test_parses_every_stats_line(tmp_path):
    rows = parse_log(_write(tmp_path))
    assert [r["iter"] for r in rows] == [1, 2, 3]
    assert rows[0]["approx_kl"] == 0.0517
    assert rows[0]["epochs"] == 1
    assert rows[2]["vmae"] == 0.680
    assert rows[2]["lr"] == 2.17e-04


def test_curve_rows_are_not_mistaken_for_stats(tmp_path):
    """The log interleaves 'iter N vs_random ...' curve rows; only 'ppo iter' lines are stats."""
    rows = parse_log(_write(tmp_path))
    assert len(rows) == 3  # not 4 -- the bare 'iter   1  vs_random' row is not a stats line


def test_summary_names_the_iters_where_the_trust_region_bound(tmp_path):
    s = summarize(parse_log(_write(tmp_path)))
    assert s["trust_region_bound_iters"] == [1]
    assert s["trust_region_bound_count"] == 1
    assert s["epochs_full_fraction"] == round(2 / 3, 4)
    assert s["approx_kl_max"] == 0.0517  # above the 0.045 bar -- which is why it bound
    assert s["kl_bar"] == 0.045


def test_reported_bar_follows_the_run_not_the_default(tmp_path):
    """Build 23 raises target_kl to 0.06. A certificate that still claimed 0.045 would name a
    bar the optimizer never enforced -- in the one artifact the build's attribution rests on."""
    s = summarize(parse_log(_write(tmp_path)), 0.09)
    assert s["kl_bar"] == 0.09
    # Binding is read off the epoch count, so it is unchanged by the bar: iter 1 still stopped
    # after 1 of 4 epochs. The bar is provenance, not the detector.
    assert s["trust_region_bound_iters"] == [1]


def test_clean_run_reports_no_bind(tmp_path):
    clean = "\n".join(line for line in LOG.splitlines() if "epochs 1" not in line)
    s = summarize(parse_log(_write(tmp_path, clean)))
    assert s["trust_region_bound_iters"] == []
    assert s["epochs_full_fraction"] == 1.0


def test_empty_log_does_not_crash(tmp_path):
    s = summarize(parse_log(_write(tmp_path, "no stats here\n")))
    assert s["n_iters"] == 0
    assert s["trust_region_bound_iters"] == []
