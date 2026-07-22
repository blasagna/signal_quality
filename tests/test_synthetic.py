"""The demo recording must contain the faults it claims to, at the times it says.

This is the notebook's worked example, so if the generator drifts the notebook
starts teaching something false. Detection is scored against the generator's own
time-resolved ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

import signal_quality as sq
from signal_quality import metrics as M

#: Faults that a *different* injected fault necessarily also produces. A dead
#: channel really does correlate with nothing, and a channel swinging to the
#: converter rail really is an amplitude outlier — these are true observations
#: about a broken channel, not false positives.
EXPECTED_SIDE_EFFECTS = {"ISOLATED", "AMP_OUTLIER", "ARTIFACT"}

#: Spectral flags are attributed through a wider analysis window, so their
#: onsets are only accurate to about that window. Amplitude flags are exact.
SPECTRAL_FLAGS = {"LINE_NOISE", "EMG"}


@pytest.fixture(scope="module")
def demo():
    rec, truth = sq.make_demo_recording()
    mf = sq.compute([M.RMS(), M.LineRatio(), M.EMGFraction(), M.MaxCorrelation(),
                     M.FlatFraction(), M.ClipFraction(), M.PeakToPeak()], rec)
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    verdicts = sq.verdict(flags, mf)
    return rec, truth, mf, flags, verdicts


def _flags_in(flags, mf, ch, t0, t1):
    """Flags on ``ch`` overlapping ``[t0, t1)``."""
    times = mf.table.groupby(level="interval")[["t_start", "t_end"]].first()
    f = flags[flags["channel"] == ch]
    if not len(f):
        return set()
    ts = times.reindex(f["interval"])
    hit = (ts["t_end"].to_numpy() > t0) & (ts["t_start"].to_numpy() < t1)
    return set(f["flag"].to_numpy()[hit])


def test_every_injected_fault_is_detected_in_its_own_window(demo):
    _, truth, mf, flags, _ = demo
    eeg = set(mf.table.index.get_level_values("channel"))
    missed = {}
    for r in truth.itertuples():
        if r.channel not in eeg:
            continue
        got = _flags_in(flags, mf, r.channel, r.t_start, r.t_end)
        if r.injected not in got:
            missed[(r.channel, round(r.t_start))] = (r.injected, got)
    assert not missed, f"undetected injected faults: {missed}"


def test_episodic_faults_are_not_flagged_outside_their_episode(demo):
    """The whole point of per-interval flagging: a fault bounded in time must
    not condemn the channel's clean stretches."""
    _, truth, mf, flags, _ = demo
    episodic = truth[(truth["t_start"] > 0) & (truth["t_end"] < 170)]
    assert len(episodic) >= 4, "generator should carry episodic faults"

    for r in episodic.itertuples():
        # A margin either side: spectral flags smear by the analysis window,
        # and an artifact's edge lands wherever the window boundary falls.
        margin = 6.0
        before = _flags_in(flags, mf, r.channel, 0, max(0, r.t_start - margin))
        after = _flags_in(flags, mf, r.channel, r.t_end + margin, 1e9)
        outside = (before | after) - EXPECTED_SIDE_EFFECTS
        assert r.injected not in outside, (
            f"{r.channel}: {r.injected} leaked outside its "
            f"{r.t_start:.0f}-{r.t_end:.0f}s episode")


def test_segments_recover_episode_timing(demo):
    _, truth, mf, _, verdicts = demo
    seg = sq.bad_segments(verdicts, mf, min_duration=2.0)

    for r in truth[truth["t_start"] > 0].itertuples():
        s = seg[(seg["channel"] == r.channel)
                & (seg["t_end"] > r.t_start) & (seg["t_start"] < r.t_end)]
        assert len(s), f"no segment covers {r.channel} {r.t_start:.0f}-{r.t_end:.0f}s"
        tol = 6.0 if r.injected in SPECTRAL_FLAGS else 2.5
        assert abs(s["t_start"].min() - r.t_start) < tol
        assert abs(s["t_end"].max() - r.t_end) < tol


def test_clean_channels_are_never_flagged(demo):
    """A demo that cries wolf on healthy channels teaches the wrong lesson."""
    _, truth, mf, flags, verdicts = demo
    eeg = set(mf.table.index.get_level_values("channel"))
    faulty = set(truth["channel"])
    clean = sorted(eeg - faulty)
    assert len(clean) > 5

    summary = sq.channel_summary(verdicts)
    assert (summary.loc[clean, "verdict"] == "good").all(), (
        summary.loc[clean][summary.loc[clean, "verdict"] != "good"])


def test_sustained_faults_dominate_their_channel(demo):
    """A fault present throughout should read as bad for most of the time, not
    as an occasional blip."""
    _, _, _, _, verdicts = demo
    s = sq.channel_summary(verdicts)
    for ch in ("T5", "C3", "F9", "T9"):
        assert s.loc[ch, "pct_bad"] > 60, (ch, s.loc[ch, "pct_bad"])


def test_metric_values_are_physically_plausible(demo):
    """Guards against a generator that 'works' only via absurd magnitudes."""
    _, truth, mf, _, _ = demo
    t = mf.table
    faulty = set(truth["channel"])
    healthy = [c for c in t.index.get_level_values("channel").unique()
               if c not in faulty]
    h = t.loc[healthy]

    assert 5 < h["rms"].median() < 60                     # µV, scalp EEG
    assert h["max_corr"].median() > 0.6
    assert h["emg_pct"].median() < 15
    line_ch = set(truth[truth["injected"] == "LINE_NOISE"]["channel"])
    bad_line = t.loc[sorted(line_ch), "line_ratio"].median()
    assert 300 < bad_line < 1e6, bad_line       # bad, but not absurd


def test_recording_level_faults(demo):
    rec, _, _, _, _ = demo
    issues = sq.check_integrity(rec)
    checks = set(issues["check"])

    assert "data_gap" in checks
    assert "nonmonotonic_time" in checks
    # OSAT is an aux channel, so it never reaches the EEG metric table; the
    # integrity pass is what catches it.
    dead = set(issues[issues["check"] == "dead_channel"]["channel"])
    assert "OSAT" in dead


def test_gap_intervals_are_no_data_not_good(demo):
    rec, _, mf, _, verdicts = demo
    blank = mf.table["coverage"] <= 0
    assert blank.any(), "demo should contain a recording gap"
    assert (verdicts.loc[blank[blank].index, "verdict"] == "no_data").all()
    assert not (verdicts["verdict"] == "good")[blank].any()


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
