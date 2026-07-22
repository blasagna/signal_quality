"""Both scales of assessment, and the rule that keeps them from double-counting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import signal_quality as sq
from signal_quality.report import _drop_episode_artifacts


@pytest.fixture(scope="module")
def demo():
    rec, truth = sq.make_demo_recording()
    return rec, truth, sq.assess(rec)


def test_sustained_and_episodic_are_both_reported(demo):
    _, truth, r = demo
    assert len(r.bad_channels) > 5
    ep = r.segments[r.segments["scope"] == "interval"]
    assert len(ep) > 5
    assert set(r.segments["scope"]) == {"channel", "interval"}


def test_episodic_channels_are_not_condemned_outright(demo):
    """The whole point: a fault bounded in time must not put the channel on the
    permanent exclusion list."""
    _, truth, r = demo
    sustained_truth = set(truth[(truth["t_start"] <= 0)]["channel"])
    episodic_only = set(truth[truth["t_start"] > 0]["channel"]) - sustained_truth
    wrongly = episodic_only & set(r.bad_channels)
    assert not wrongly, f"episodic-only channels condemned as sustained: {wrongly}"


def test_sustained_channels_are_found(demo):
    _, truth, r = demo
    eeg = set(r.metrics.table.index.get_level_values("channel"))
    sustained_truth = {c for c in truth[truth["t_start"] <= 0]["channel"] if c in eeg}
    # F7/F8 carry EMG, which is a marginal-severity flag by design and so is
    # not part of the bad-segment output.
    expected = sustained_truth - {"F7", "F8"}
    assert expected <= set(r.bad_channels), expected - set(r.bad_channels)


def test_episodes_land_on_the_right_spans(demo):
    _, truth, r = demo
    ep = r.segments[r.segments["scope"] == "interval"]
    for row in truth[truth["t_start"] > 0].itertuples():
        s = ep[
            (ep["channel"] == row.channel)
            & (ep["t_end"] > row.t_start)
            & (ep["t_start"] < row.t_end)
        ]
        assert len(s), f"no episode for {row.channel} at {row.t_start:.0f}s"


# --- the drop rule -----------------------------------------------------------


def _seg(ch, t0, t1, reasons="FLAT"):
    return dict(
        channel=ch,
        t_start=t0,
        t_end=t1,
        duration=t1 - t0,
        severity="bad",
        reasons=reasons,
        n_intervals=int(t1 - t0),
    )


def test_a_concentrated_substantial_episode_explains_the_channel_flag():
    """One big artifact can shift a whole-recording statistic enough to condemn
    an otherwise fine channel; the episode already describes it better."""
    sustained = pd.DataFrame([_seg("C3", 0, 180)])
    episodes = pd.DataFrame([_seg("C3", 60, 80)])  # 11% of 180 s
    out = _drop_episode_artifacts(sustained, episodes, duration=180.0)
    assert not len(out)


def test_a_tiny_episode_cannot_explain_a_channel_flag():
    """The reference study's purely-sustained channels had episodes covering
    0.1% of the recording — far too little to move a whole-recording number."""
    sustained = pd.DataFrame([_seg("C3", 0, 180)])
    episodes = pd.DataFrame([_seg("C3", 60, 60.2)])  # 0.1%
    out = _drop_episode_artifacts(sustained, episodes, duration=180.0)
    assert len(out) == 1


def test_scattered_episodes_keep_the_channel_flag():
    """A sustained defect that only trips per-second thresholds intermittently
    still fails seconds spread across the whole recording."""
    sustained = pd.DataFrame([_seg("A1", 0, 180)])
    episodes = pd.DataFrame([_seg("A1", 5, 12), _seg("A1", 90, 96), _seg("A1", 160, 172)])
    out = _drop_episode_artifacts(sustained, episodes, duration=180.0)
    assert len(out) == 1


def test_flag_names_need_not_match_across_scales():
    """The same event trips different metrics at different scales — a movement
    episode reads ARTIFACT per second but ISOLATED over the recording. Requiring
    matching labels would keep exactly the findings this removes."""
    sustained = pd.DataFrame([_seg("Pz", 0, 180, reasons="ISOLATED")])
    episodes = pd.DataFrame([_seg("Pz", 36, 51, reasons="AMP_OUTLIER+ARTIFACT")])
    out = _drop_episode_artifacts(sustained, episodes, duration=180.0)
    assert not len(out)


def test_channel_with_no_episodes_is_always_kept():
    sustained = pd.DataFrame([_seg("C3", 0, 180)])
    out = _drop_episode_artifacts(
        sustained, pd.DataFrame(columns=sustained.columns), duration=180.0
    )
    assert len(out) == 1


# --- report shape ------------------------------------------------------------


def test_excluded_time_covers_both_scopes(demo):
    _, _, r = demo
    ex = r.excluded_time()
    assert (ex > 0).all()
    assert ex.index.isin(r.segments["channel"]).all()


def test_report_exposes_the_pieces(demo):
    rec, _, r = demo
    assert r.metrics.table.index.names == ["channel", "interval"]
    assert list(r.verdicts.index.names) == ["channel", "interval"]
    assert "pct_bad" in r.channels.columns
    assert set(r.issues.columns) >= {"check", "severity", "detail"}
    assert "scope" in r.segments.columns


def test_assess_respects_a_custom_window():
    rec, _ = sq.make_demo_recording(duration=60.0)
    r = sq.assess(rec, window=5.0)
    widths = r.metrics.table["t_end"] - r.metrics.table["t_start"]
    assert np.allclose(widths.to_numpy(), 5.0)
