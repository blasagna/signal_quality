"""Running both scales of assessment, and combining them.

Quality faults come at two scales and neither subsumes the other:

* **Channel-scope, sustained.** An attenuated electrode or one that shares no
  signal with the montage is abnormal *relative to the other channels over the
  whole recording*. Per-second this is nearly invisible: on the reference study
  an electrode attenuated to a third of its neighbours scores a robust z of
  −8.9 over the recording but a median of only −1.2 per second, because the
  spread across channels within any one second is far wider than the spread of
  their whole-recording averages. Judged only per-second, that electrode passes.

* **Time-scope, episodic.** Movement, a lead pulled loose, a transient
  saturation. These are invisible to a whole-recording summary, which averages
  them away into a channel that looks merely mediocre.

So :func:`assess` runs both and labels each finding with its ``scope``. On the
reference study the sustained pass reproduces the known 13-channel exclusion
list exactly, while the interval pass adds 340 time-localized segments that a
channel-level verdict could not express.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import pandas as pd

from .core.intervals import IntervalGrid
from .core.metric import compute
from .filters import (
    DEFAULT_FILTERS,
    WHOLE_RECORDING_FILTERS,
    apply_filters,
    bad_segments,
    channel_summary,
    verdict,
)
from .metrics import DEFAULT_METRICS
from .metrics.integrity import check_integrity


@dataclass
class QualityReport:
    """Everything the two passes produced, kept separable rather than merged."""

    recording: object
    issues: pd.DataFrame  # recording-scope: gaps, clock, alignment
    metrics: object  # per-interval MetricFrame
    flags: pd.DataFrame
    verdicts: pd.DataFrame  # per (channel, interval)
    channels: pd.DataFrame  # channel_summary of the above
    segments: pd.DataFrame  # combined, with a `scope` column
    sustained: pd.DataFrame = field(repr=False, default=None)
    whole: object = field(repr=False, default=None)

    @property
    def bad_channels(self) -> list:
        """Channels with a sustained, channel-scope defect."""
        if self.sustained is None or not len(self.sustained):
            return []
        return sorted(set(self.sustained["channel"]))

    def excluded_time(self) -> pd.Series:
        """Seconds excluded per channel, across both scopes."""
        if not len(self.segments):
            return pd.Series(dtype=float)
        return self.segments.groupby("channel")["duration"].sum().sort_values(ascending=False)

    def __repr__(self) -> str:
        n_ep = int((self.segments["scope"] == "interval").sum()) if len(self.segments) else 0
        return (
            f"<QualityReport {len(self.channels)} channels, "
            f"{len(self.bad_channels)} with sustained defects, "
            f"{n_ep} time-localized segments>"
        )


def _drop_episode_artifacts(sustained, episodes, duration, min_span=0.5, min_episode_frac=0.01):
    """Remove channel-scope findings that are really one episode in disguise.

    A whole-recording summary is not immune to episodes: a large enough
    artifact shifts a channel's whole-recording statistics until the
    channel-scope filters fire, and the channel is then condemned outright even
    though it was fine either side. Calling that "sustained" would undo the
    point of flagging per interval.

    A channel-scope finding is discarded when the channel's problems are both:

    1. **Concentrated** — the episodes span less than ``min_span`` of the
       recording. A genuinely sustained defect fails seconds scattered
       throughout, even when it only trips the per-second thresholds
       intermittently.
    2. **Substantial** — the episodes cover at least ``min_episode_frac`` of the
       recording. This is the condition that carries most of the weight: on the
       reference study the two channels with purely sustained defects had
       episodes covering **0.1%** of the recording, far too little to move a
       whole-recording statistic, while every episodic artifact covered 3-15%.

    Deliberately *not* required: that the two scales agree on the flag name. The
    same physical event trips different metrics at different scales — a movement
    episode reads ``ARTIFACT`` per second but decorrelates the channel enough to
    read ``ISOLATED`` over the recording — so demanding matching labels keeps
    exactly the findings this is meant to remove.
    """
    if not len(sustained) or not len(episodes) or not duration:
        return sustained

    drop = []
    for ch, _rows in sustained.groupby("channel"):
        ep = episodes[episodes["channel"] == ch]
        if not len(ep):
            continue  # nothing to explain it
        span = (ep["t_end"].max() - ep["t_start"].min()) / duration
        frac = ep["duration"].sum() / duration
        if span < min_span and frac >= min_episode_frac:
            drop.append(ch)
    return sustained[~sustained["channel"].isin(drop)]


def assess(
    rec,
    metrics=None,
    window: float | None = None,
    filters=None,
    whole_filters=None,
    min_duration: float = 2.0,
    ch_type: str | None = "eeg",
) -> QualityReport:
    """Run the full quality assessment at both scales.

    ``window`` defaults to 1-second non-overlapping intervals.
    ``min_duration`` drops sub-2-second episodes, which are mostly threshold
    flicker rather than real events — a channel sitting near a cut-off will
    cross it repeatedly from one second to the next.
    """
    mets = [
        m() if isinstance(m, type) else m
        for m in (metrics if metrics is not None else DEFAULT_METRICS)
    ]
    grid = IntervalGrid.fixed(rec, window or IntervalGrid.DEFAULT_WINDOW)

    mf = compute(mets, rec, grid, ch_type=ch_type)
    flags = apply_filters(mf, filters if filters is not None else DEFAULT_FILTERS)
    verdicts = verdict(flags, mf)
    episodes = bad_segments(verdicts, mf, min_duration=min_duration)

    # The channel-scope pass genuinely recomputes over the whole recording
    # rather than aggregating the per-interval table. Aggregating is tempting —
    # it is free — but the median of 1-second values does not have the same
    # distribution as a whole-recording estimate (a clean channel's median
    # line_ratio over 1 s windows runs 5-10x its whole-recording value), so the
    # calibrated channel-scope thresholds no longer apply and the pass both
    # misses real defects and invents new ones. Verified: recomputing reproduces
    # the reference study's known 13-channel list exactly; aggregating missed 4
    # and added 3.
    gc.collect()
    mets_w = [
        m() if isinstance(m, type) else m
        for m in (metrics if metrics is not None else DEFAULT_METRICS)
    ]
    whole = compute(mets_w, rec, IntervalGrid.whole(rec), ch_type=ch_type)
    flags_w = apply_filters(
        whole, whole_filters if whole_filters is not None else WHOLE_RECORDING_FILTERS
    )
    sustained = bad_segments(verdict(flags_w, whole), whole)

    sustained = _drop_episode_artifacts(sustained, episodes, rec.duration)

    segments = (
        pd.concat(
            [sustained.assign(scope="channel"), episodes.assign(scope="interval")],
            ignore_index=True,
        )
        if len(sustained) or len(episodes)
        else episodes.assign(scope="interval")
    )

    return QualityReport(
        recording=rec,
        issues=check_integrity(rec),
        metrics=mf,
        flags=flags,
        verdicts=verdicts,
        channels=channel_summary(verdicts),
        segments=segments,
        sustained=sustained,
        whole=whole,
    )
