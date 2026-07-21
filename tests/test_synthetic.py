"""The demo recording must actually contain the faults it claims to.

This is the notebook's worked example, so if the generator drifts the notebook
starts teaching something false. These tests score detection against the
generator's own ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

import signal_quality as sq
from signal_quality import metrics as M

#: Faults that a *different* injected fault necessarily also produces. A dead
#: channel really does correlate with nothing, and a channel swinging to the
#: converter rail really is uncorrelated with the montage — these are true
#: observations about a broken channel, not false positives.
EXPECTED_SIDE_EFFECTS = {"T5": {"ISOLATED"}, "Fpz": {"ISOLATED"}}


@pytest.fixture(scope="module")
def demo():
    rec, truth = sq.make_demo_recording()
    mf = sq.compute([M.RMS(), M.LineRatio(), M.EMGFraction(), M.MaxCorrelation(),
                     M.FlatFraction(), M.ClipFraction()],
                    rec, sq.IntervalGrid.whole(rec))
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    eeg = mf.table.index.get_level_values("channel").unique()
    return rec, truth, mf, flags, sq.verdict(flags, channels=eeg)


def _detected(flags, ch):
    return set(flags[flags["channel"] == ch]["flag"])


def test_every_injected_fault_is_detected(demo):
    _, truth, mf, flags, _ = demo
    eeg = set(mf.table.index.get_level_values("channel"))
    missed = {}
    for ch, row in truth.iterrows():
        if not row["injected"] or ch not in eeg:
            continue
        want = set(row["injected"].split("+"))
        got = _detected(flags, ch)
        if not want <= got:
            missed[ch] = (want - got, got)
    assert not missed, f"undetected injected faults: {missed}"


def test_clean_channels_are_not_flagged(demo):
    """A demo that cries wolf on healthy channels teaches the wrong lesson."""
    _, truth, mf, flags, verdicts = demo
    eeg = set(mf.table.index.get_level_values("channel"))
    clean = [c for c in eeg if not truth.loc[c, "injected"]]
    noisy = {c: _detected(flags, c) for c in clean if _detected(flags, c)}
    assert not noisy, f"clean channels flagged: {noisy}"
    assert (verdicts.loc[clean, "verdict"] == "good").all()


def test_no_unexplained_extra_flags(demo):
    _, truth, mf, flags, _ = demo
    eeg = set(mf.table.index.get_level_values("channel"))
    for ch in eeg:
        want = set(truth.loc[ch, "injected"].split("+")) - {""}
        extra = _detected(flags, ch) - want - EXPECTED_SIDE_EFFECTS.get(ch, set())
        assert not extra, f"{ch} picked up unexplained flags: {extra}"


def test_metric_values_are_physically_plausible(demo):
    """Guards against a generator that 'works' only via absurd magnitudes."""
    _, truth, mf, _, _ = demo
    t = mf.table.droplevel("interval")
    healthy = [c for c in t.index if not truth.loc[c, "injected"]]

    assert 5 < t.loc[healthy, "rms"].median() < 60          # µV, scalp EEG
    assert t.loc[healthy, "max_corr"].min() > 0.6           # shared reference
    assert t.loc[healthy, "emg_pct"].max() < 10
    line = t.loc[[c for c in t.index if "LINE_NOISE" in truth.loc[c, "injected"]],
                 "line_ratio"]
    assert (line > 300).all() and (line < 1e5).all()        # bad, not absurd
    assert 0 < t.loc["Fpz", "clip_pct"] < 1.0               # brief saturation


def test_recording_level_faults(demo):
    rec, _, _, _, _ = demo
    issues = sq.check_integrity(rec)
    checks = set(issues["check"])

    assert "data_gap" in checks
    assert "nonmonotonic_time" in checks
    # OSAT is an aux channel, so it never reaches the EEG metric table; the
    # integrity pass is what catches it.
    dead = set(issues[issues["check"] == "dead_channel"]["channel"])
    assert {"OSAT", "T5"} <= dead


def test_gap_is_excluded_not_analysed(demo):
    rec, _, _, _, _ = demo
    assert 0.05 < 1 - rec.covered.mean() < 0.3
    grid = sq.IntervalGrid.fixed(rec, 10.0)
    assert (grid.table["coverage"] < 1.0).any()


def test_counts_present_so_clipping_is_measurable(demo):
    rec, _, mf, _, _ = demo
    assert rec.has_counts
    assert not mf.table["clip_pct"].isna().all()
    assert np.abs(rec.ds["counts"].values).max() <= 131071


def test_is_reproducible_and_seed_varies():
    a, _ = sq.make_demo_recording(seed=1)
    b, _ = sq.make_demo_recording(seed=1)
    c, _ = sq.make_demo_recording(seed=2)
    np.testing.assert_array_equal(a.ds["signal"].values, b.ds["signal"].values)
    assert not np.array_equal(a.ds["signal"].values, c.ds["signal"].values)


def test_channels_can_be_placed_on_the_head():
    """The demo must work with the topomap, or it cannot show the head plots."""
    rec, _ = sq.make_demo_recording()
    placed, xy, unplaced = sq.montage.place(rec.pick(ch_type="eeg").ch_names)
    assert len(placed) >= 25
    assert np.isfinite(xy).all()
