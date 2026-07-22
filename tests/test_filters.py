"""Filters are policy applied to a finished table, independent of the metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import signal_quality as sq
from signal_quality import metrics as M


@pytest.fixture
def whole(faulty_rec):
    """Whole-recording frame, judged by the whole-recording thresholds."""
    mf = sq.compute(
        [M.RMS(), M.LineRatio(), M.FlatFraction(), M.MaxCorrelation(), M.EMGFraction()],
        faulty_rec,
        sq.IntervalGrid.whole(faulty_rec),
    )
    return mf, sq.apply_filters(mf, sq.WHOLE_RECORDING_FILTERS)


def test_whole_recording_filters_catch_each_injected_fault(whole):
    _, flags = whole
    fired = {(r["channel"], r["flag"]) for _, r in flags.iterrows()}

    assert ("FLAT", "FLAT") in fired
    assert ("FLAT", "ISOLATED") in fired
    assert ("LINE", "LINE_NOISE") in fired
    assert ("LOUD", "AMP_OUTLIER") in fired
    assert not any(ch == "C4" for ch, _ in fired)


def test_defaults_do_not_flag_bridging(whole):
    """Absolute correlation cannot separate a bridge from a normal neighbour on
    a shared reference, so no default filter may claim to."""
    _, flags = whole
    assert "BRIDGED" not in set(flags["flag"])
    assert "BRIDGED" not in {f.flag for f in sq.DEFAULT_FILTERS}


def test_correlation_pairs_shortlists_the_bridge(faulty_rec):
    """The diagnostic tool still names the pair, for a human to adjudicate."""
    pairs = M.correlation_pairs(faulty_rec, threshold=0.95)
    assert set(pairs.iloc[0][["channel_a", "channel_b"]]) == {"C3", "BRIDGE"}
    assert pairs["elec_distance"].idxmin() == 0


def test_thresholds_are_policy_not_recomputation(whole):
    """The same table judged by two policies yields different verdicts."""
    mf, _ = whole
    strict = sq.apply_filters(
        mf, [sq.Threshold(metric="line_ratio", op=">", value=2, flag="LINE_NOISE")]
    )
    lenient = sq.apply_filters(
        mf, [sq.Threshold(metric="line_ratio", op=">", value=1e9, flag="LINE_NOISE")]
    )
    assert len(strict) > 0
    assert len(lenient) == 0


def test_robust_z_is_resistant_to_the_outlier_it_finds():
    """A mean/std z-score would be dragged out by the outlier; MAD is not."""
    vals = [10.0] * 20 + [1000.0]
    idx = pd.MultiIndex.from_product(
        [[f"c{i}" for i in range(21)], [0]], names=["channel", "interval"]
    )
    table = pd.DataFrame({"rms": vals}, index=idx)
    z = sq.RobustZ.zscores(table, "rms")
    assert abs(z.iloc[-1]) > 10
    assert z.iloc[:-1].abs().max() < 1e-6


def test_filter_on_missing_column_is_silent(whole):
    """A metric that was not computed must not raise or fabricate flags."""
    mf, _ = whole
    out = sq.apply_filters(mf, [sq.Threshold(metric="clip_pct", op=">", value=0, flag="CLIPPING")])
    assert len(out) == 0


def test_flags_carry_the_evidence(whole):
    _, flags = whole
    assert set(sq.filters.FLAG_COLUMNS) <= set(flags.columns)
    line = flags[(flags["flag"] == "LINE_NOISE") & (flags["channel"] == "LINE")]
    assert len(line)
    assert line["value"].iloc[0] > line["threshold"].iloc[0]


# --- per-interval verdicts ---------------------------------------------------


@pytest.fixture
def per_interval(faulty_rec):
    mf = sq.compute([M.RMS(), M.FlatFraction(), M.MaxCorrelation(), M.PeakToPeak()], faulty_rec)
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    return mf, flags, sq.verdict(flags, mf)


def test_verdict_is_indexed_by_channel_and_interval(per_interval):
    mf, _, v = per_interval
    assert list(v.index.names) == ["channel", "interval"]
    assert v.index.equals(mf.table.index)
    assert set(v["verdict"]) <= set(sq.filters.SEVERITY_ORDER)


def test_verdict_marks_unflagged_cells_good_not_missing(per_interval):
    """Without the metric frame only flagged cells would appear, and a clean
    stretch would be indistinguishable from one never assessed."""
    _, flags, v = per_interval
    assert (v["verdict"] == "good").any()
    assert len(v) > len(flags)


def test_verdict_takes_the_worst_severity_in_a_cell():
    flags = pd.DataFrame(
        [
            ("C3", 0, "EMG", "marginal", "emg_pct", 40.0, 35.0),
            ("C3", 0, "FLAT", "bad", "flat_frac", 1.0, 0.5),
        ],
        columns=sq.filters.FLAG_COLUMNS,
    )
    v = sq.verdict(flags)
    assert v.loc[("C3", 0), "verdict"] == "bad"
    assert v.loc[("C3", 0), "reasons"] == "FLAT"  # worst-severity flags only
    assert v.loc[("C3", 0), "n_flags"] == 2


def test_channel_summary_percentages_use_covered_time(gapped_rec):
    mf = sq.compute([M.RMS(), M.PeakToPeak()], gapped_rec)
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    v = sq.verdict(flags, mf)
    s = sq.channel_summary(v)

    assert (s["pct_no_data"] > 0).all()  # the fixture has a gap
    # good+marginal+bad must account for all *covered* time, gap excluded.
    total = s["pct_good"] + s["pct_marginal"] + s["pct_bad"]
    np.testing.assert_allclose(total.to_numpy(), 100.0, atol=1e-9)


def test_channel_summary_needs_sustained_badness(per_interval):
    """At 1-second granularity almost every channel has some bad second, so
    'any bad interval condemns the channel' would condemn the montage."""
    _, _, v = per_interval
    strict = sq.channel_summary(v, bad_time_frac=0.0)
    lenient = sq.channel_summary(v, bad_time_frac=0.95)
    assert (strict["verdict"] == "bad").sum() >= (lenient["verdict"] == "bad").sum()


# --- segments ----------------------------------------------------------------


def _verdicts(rows, n_ch=1):
    idx = pd.MultiIndex.from_tuples([r[:2] for r in rows], names=["channel", "interval"])
    return pd.DataFrame(
        {"verdict": [r[2] for r in rows], "reasons": [r[3] for r in rows], "n_flags": 1}, index=idx
    )


def test_segments_merge_contiguous_intervals():
    v = _verdicts([("C3", i, "bad", "FLAT") for i in range(5)])
    seg = sq.bad_segments(v)
    assert len(seg) == 1
    assert seg.iloc[0]["n_intervals"] == 5
    assert seg.iloc[0]["duration"] == pytest.approx(5.0)


def test_segments_split_on_a_real_gap():
    rows = [("C3", i, "bad", "FLAT") for i in (0, 1, 2, 20, 21)]
    seg = sq.bad_segments(_verdicts(rows))
    assert len(seg) == 2
    assert seg.iloc[0]["n_intervals"] == 3
    assert seg.iloc[1]["n_intervals"] == 2


def test_segments_respect_min_duration():
    rows = [("C3", i, "bad", "FLAT") for i in (0, 1, 2, 3, 4, 30)]
    assert len(sq.bad_segments(_verdicts(rows))) == 2
    assert len(sq.bad_segments(_verdicts(rows), min_duration=2.0)) == 1


def test_segments_are_empty_when_nothing_is_bad():
    v = _verdicts([("C3", i, "good", "") for i in range(5)])
    seg = sq.bad_segments(v)
    assert len(seg) == 0
    assert "t_start" in seg.columns  # still correctly shaped


def test_annotations_are_channel_scoped(per_interval):
    """Channel scoping is the point: a bad stretch on one electrode must not
    exclude the whole montage for that epoch."""
    mf, _, v = per_interval
    seg = sq.bad_segments(v, mf)
    ann = sq.to_annotations(seg)
    assert len(ann) == len(seg)
    if len(seg):
        assert all(len(c) == 1 for c in ann.ch_names)
        assert all(d.startswith("BAD_") for d in ann.description)
