"""Filters are policy applied to a finished table, independent of the metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import signal_quality as sq
from signal_quality import metrics as M


@pytest.fixture
def mf(faulty_rec):
    return sq.compute([M.RMS(), M.LineRatio(), M.FlatFraction(),
                       M.MaxCorrelation(), M.EMGFraction()],
                      faulty_rec, sq.IntervalGrid.whole(faulty_rec))


def test_default_filters_catch_each_injected_fault(mf, faulty_rec):
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    fired = {(r["channel"], r["flag"]) for _, r in flags.iterrows()}

    assert ("FLAT", "FLAT") in fired
    assert ("FLAT", "ISOLATED") in fired
    assert ("LINE", "LINE_NOISE") in fired
    assert ("LOUD", "AMP_OUTLIER") in fired
    assert not any(ch == "C4" for ch, _ in fired)


def test_defaults_do_not_flag_bridging(mf):
    """Absolute correlation cannot separate a bridge from a normal neighbour on
    a shared reference, so no default filter may claim to."""
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    assert "BRIDGED" not in set(flags["flag"])


def test_correlation_pairs_shortlists_the_bridge(faulty_rec):
    """The diagnostic tool still names the pair, for a human to adjudicate."""
    pairs = M.correlation_pairs(faulty_rec, threshold=0.95)
    assert set(pairs.iloc[0][["channel_a", "channel_b"]]) == {"C3", "BRIDGE"}
    # The bridged pair also has by far the smallest electrical distance.
    assert pairs["elec_distance"].idxmin() == 0


def test_verdict_aggregates_to_worst_severity(mf, faulty_rec):
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    v = sq.verdict(flags, channels=faulty_rec.ch_names)

    assert v.loc["C4", "verdict"] == "good"
    assert v.loc["FLAT", "verdict"] == "bad"
    assert v.loc["LINE", "verdict"] == "bad"
    assert "LINE_NOISE" in v.loc["LINE", "reasons"]


def test_thresholds_are_policy_not_recomputation(mf):
    """The same table judged by two policies yields different verdicts."""
    strict = sq.apply_filters(mf, [sq.Threshold(metric="line_ratio", op=">",
                                                value=2, flag="LINE_NOISE")])
    lenient = sq.apply_filters(mf, [sq.Threshold(metric="line_ratio", op=">",
                                                 value=1e9, flag="LINE_NOISE")])
    assert len(strict) > 0
    assert len(lenient) == 0


def test_robust_z_is_resistant_to_the_outlier_it_finds():
    """A mean/std z-score would be dragged out by the outlier; MAD is not."""
    vals = [10.0] * 20 + [1000.0]
    idx = pd.MultiIndex.from_product([[f"c{i}" for i in range(21)], [0]],
                                     names=["channel", "interval"])
    table = pd.DataFrame({"rms": vals}, index=idx)
    z = sq.RobustZ.zscores(table, "rms")
    assert abs(z.iloc[-1]) > 10
    assert z.iloc[:-1].abs().max() < 1e-6


def test_filter_on_missing_column_is_silent(mf):
    """A metric that was not computed must not raise or fabricate flags."""
    out = sq.apply_filters(mf, [sq.Threshold(metric="clip_pct", op=">", value=0,
                                             flag="CLIPPING")])
    assert len(out) == 0


def test_flags_carry_the_evidence(mf):
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    assert set(sq.filters.FLAG_COLUMNS) <= set(flags.columns)
    line = flags[(flags["flag"] == "LINE_NOISE") & (flags["channel"] == "LINE")]
    assert len(line)
    assert line["value"].iloc[0] > line["threshold"].iloc[0]
