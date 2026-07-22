"""Filters: the policy layer.

Metrics measure; filters decide. Keeping them separate means the same computed
table can be judged by a strict policy and a lenient one without recomputing
anything, and that a threshold change is a one-line edit rather than a hunt
through analysis code.

A filter reads the joined metric table and emits rows into a long flags table::

    channel | interval | flag | severity | metric | value | threshold

Note that ``RobustZ`` is a filter rather than a metric on purpose: "this channel
is an amplitude outlier" is a statement about a channel *relative to its peers*,
which is a comparison across rows of the finished table, not a measurement of
the channel itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FLAG_COLUMNS = ["channel", "interval", "flag", "severity", "metric", "value", "threshold"]

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


@dataclass
class Filter:
    """Base class. Subclasses implement :meth:`apply`."""

    flag: str = "FLAG"
    severity: str = "bad"

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _emit(self, table, mask, metric, values, threshold) -> pd.DataFrame:
        idx = table.index[mask]
        if not len(idx):
            return pd.DataFrame(columns=FLAG_COLUMNS)
        return pd.DataFrame(
            {
                "channel": idx.get_level_values("channel"),
                "interval": idx.get_level_values("interval"),
                "flag": self.flag,
                "severity": self.severity,
                "metric": metric,
                "value": np.asarray(values)[mask.to_numpy()],
                "threshold": threshold,
            }
        )


@dataclass
class Threshold(Filter):
    """Flag rows where a metric crosses an absolute threshold."""

    metric: str = ""
    op: str = ">"
    value: float = 0.0
    flag: str = "FLAG"
    severity: str = "bad"

    def apply(self, table):
        if self.metric not in table.columns:
            return pd.DataFrame(columns=FLAG_COLUMNS)
        col = table[self.metric]
        mask = _OPS[self.op](col, self.value) & col.notna()
        return self._emit(table, mask, self.metric, col.to_numpy(), self.value)


@dataclass
class RobustZ(Filter):
    """Flag rows whose metric is an outlier *relative to the other channels*.

    Uses the median/MAD z-score (``0.6745 * (x - median) / MAD``), which unlike a
    mean/std z-score is not dragged around by the very outliers it is meant to
    find. Applied per interval, so a drifting recording is judged against its
    contemporaneous peers.

    ``log=True`` compares in log space, appropriate for amplitude-like
    quantities whose spread is multiplicative.
    """

    metric: str = ""
    abs_gt: float | None = None
    gt: float | None = None
    lt: float | None = None
    log: bool = True
    flag: str = "OUTLIER"
    severity: str = "bad"

    def apply(self, table):
        if self.metric not in table.columns:
            return pd.DataFrame(columns=FLAG_COLUMNS)
        z = self.zscores(table, self.metric, log=self.log)
        mask = pd.Series(False, index=table.index)
        if self.abs_gt is not None:
            mask |= z.abs() > self.abs_gt
        if self.gt is not None:
            mask |= z > self.gt
        if self.lt is not None:
            mask |= z < self.lt
        mask &= z.notna()
        thr = (
            self.abs_gt
            if self.abs_gt is not None
            else (self.gt if self.gt is not None else self.lt)
        )
        return self._emit(table, mask, self.metric, z.to_numpy(), thr)

    @staticmethod
    def zscores(table: pd.DataFrame, metric: str, log: bool = True) -> pd.Series:
        """Median/MAD z-score of ``metric``, computed within each interval."""
        col = table[metric].astype(float)
        if log:
            col = np.log(col.where(col > 0))

        def _z(s):
            med = s.median()
            mad = (s - med).abs().median()
            return 0.6745 * (s - med) / (mad + 1e-9)

        return col.groupby(level="interval").transform(_z)


def apply_filters(mf, filters) -> pd.DataFrame:
    """Apply every filter to the joined table, returning the long flags table.

    Accepts a :class:`~signal_quality.core.frame.MetricFrame` or a bare
    DataFrame.
    """
    table = getattr(mf, "table", mf)
    out = [f.apply(table) for f in filters]
    out = [f for f in out if len(f)]
    if not out:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    return pd.concat(out, ignore_index=True).sort_values(
        ["channel", "interval", "severity"], ignore_index=True
    )


#: Verdict categories, worst last. ``no_data`` is deliberately *not* an
#: ordinary severity: an interval inside a recording gap has nothing to judge,
#: and calling it "good" would count missing data as quality.
SEVERITY_ORDER = {"no_data": -1, "good": 0, "marginal": 1, "bad": 2}


def verdict(flags: pd.DataFrame, mf=None, channels=None) -> pd.DataFrame:
    """Judge every ``(channel, interval)`` cell.

    Returns a frame indexed by ``(channel, interval)`` with ``verdict``
    (``no_data``/``good``/``marginal``/``bad``), ``reasons`` — the flags that
    fired at the worst severity — and ``n_flags``.

    ``mf`` is the metric frame the flags came from. Pass it: without it, only
    cells that were flagged appear, so a clean recording looks empty and there
    is nothing to distinguish "assessed and fine" from "never assessed".
    """
    if mf is not None:
        table = getattr(mf, "table", mf)
        index = table.index
        coverage = table["coverage"] if "coverage" in table.columns else None
    elif len(flags):
        index = pd.MultiIndex.from_frame(
            flags[["channel", "interval"]].drop_duplicates(), names=["channel", "interval"]
        )
        coverage = None
    else:
        return pd.DataFrame(columns=["verdict", "reasons", "n_flags"])

    if channels is not None:
        keep = set(channels)
        index = index[index.get_level_values("channel").isin(keep)]
        if coverage is not None:
            coverage = coverage.loc[index]

    out = pd.DataFrame(index=index)
    out["verdict"] = "good"
    out["reasons"] = ""
    out["n_flags"] = 0

    if len(flags):
        f = flags.set_index(["channel", "interval"])
        f = f[f.index.isin(index)]
        if len(f):
            rank = f["severity"].map(SEVERITY_ORDER).fillna(0)
            worst = rank.groupby(level=[0, 1]).max()
            out.loc[worst.index, "verdict"] = worst.map({v: k for k, v in SEVERITY_ORDER.items()})
            top = f[rank == rank.groupby(level=[0, 1]).transform("max")]
            reasons = top.groupby(level=[0, 1])["flag"].agg(lambda s: "+".join(sorted(set(s))))
            out.loc[reasons.index, "reasons"] = reasons
            counts = f.groupby(level=[0, 1]).size()
            out.loc[counts.index, "n_flags"] = counts

    if coverage is not None:
        blank = coverage.reindex(out.index) <= 0
        out.loc[blank, ["verdict", "reasons", "n_flags"]] = ["no_data", "", 0]

    return out.sort_index()


def channel_summary(
    verdicts: pd.DataFrame,
    bad_time_frac: float = 0.20,
    marginal_time_frac: float = 0.20,
    top_reasons: int = 3,
) -> pd.DataFrame:
    """Roll per-interval verdicts up to one row per channel.

    Percentages are over **covered** time only, so a channel is not rewarded for
    a recording gap. ``pct_bad`` is the useful continuous quantity — far more
    informative than a binary label, and what the topomap should show.

    The ``verdict`` column applies a time threshold: a channel counts as bad
    when it is bad for more than ``bad_time_frac`` of its covered time. At
    1-second granularity almost every channel has *some* bad second, so
    "any bad interval condemns the channel" would condemn the whole montage.
    """
    if not len(verdicts):
        return pd.DataFrame(
            columns=["pct_bad", "pct_marginal", "pct_good", "pct_no_data", "n_intervals", "verdict"]
        )

    g = verdicts.groupby(level="channel")["verdict"]
    counts = g.value_counts().unstack(fill_value=0)
    for c in ("good", "marginal", "bad", "no_data"):
        if c not in counts:
            counts[c] = 0

    total = counts[["good", "marginal", "bad", "no_data"]].sum(axis=1)
    covered = total - counts["no_data"]
    denom = covered.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "pct_bad": 100 * counts["bad"] / denom,
            "pct_marginal": 100 * counts["marginal"] / denom,
            "pct_good": 100 * counts["good"] / denom,
            "pct_no_data": 100 * counts["no_data"] / total.replace(0, np.nan),
            "n_intervals": total,
        }
    )
    out["verdict"] = np.where(
        out["pct_bad"].isna(),
        "no_data",
        np.where(
            out["pct_bad"] > 100 * bad_time_frac,
            "bad",
            np.where(
                out["pct_bad"] + out["pct_marginal"] > 100 * marginal_time_frac, "marginal", "good"
            ),
        ),
    )

    # Dominant reasons, ranked by how often each fired. A union of every flag
    # that ever fired would list almost everything for almost every channel and
    # say nothing about what is actually wrong.
    exploded = verdicts[verdicts["reasons"] != ""]["reasons"].str.split("+").explode()
    if len(exploded):
        reason_counts = exploded.groupby(level="channel").value_counts()
        reasons = reason_counts.groupby(level="channel").apply(
            lambda s: "+".join(s.head(top_reasons).index.get_level_values(-1))
        )
        out["reasons"] = reasons.reindex(out.index).fillna("")
    else:
        out["reasons"] = ""
    return out.sort_values("pct_bad", ascending=False)


def bad_segments(
    verdicts: pd.DataFrame,
    mf=None,
    severities=("bad",),
    merge_gap: float = 1.0,
    min_duration: float = 0.0,
) -> pd.DataFrame:
    """Contiguous runs of flagged intervals, per channel, as time spans.

    This is the deliverable: *exclude C3 from 412 s to 438 s*, rather than
    condemning C3 outright. Runs separated by no more than ``merge_gap`` seconds
    are joined, so one artifact episode is one row instead of dozens of
    single-second fragments.

    Returns ``(channel, t_start, t_end, duration, severity, reasons,
    n_intervals)``.
    """
    cols = ["channel", "t_start", "t_end", "duration", "severity", "reasons", "n_intervals"]
    if not len(verdicts):
        return pd.DataFrame(columns=cols)

    times = None
    if mf is not None:
        table = getattr(mf, "table", mf)
        times = table[["t_start", "t_end"]].droplevel("channel").groupby(level="interval").first()

    hit = verdicts[verdicts["verdict"].isin(severities)]
    if not len(hit):
        return pd.DataFrame(columns=cols)

    rows = []
    for ch, sub in hit.groupby(level="channel"):
        ivs = sorted(sub.index.get_level_values("interval"))
        run = [ivs[0]]
        for prev, cur in zip(ivs, ivs[1:], strict=False):
            t_prev_end = _t(times, prev, "t_end")
            t_cur_start = _t(times, cur, "t_start")
            contiguous = (t_cur_start - t_prev_end) <= merge_gap
            if contiguous:
                run.append(cur)
            else:
                rows.append(_segment(ch, run, sub, times))
                run = [cur]
        rows.append(_segment(ch, run, sub, times))

    out = pd.DataFrame(rows, columns=cols)
    out = out[out["duration"] >= min_duration]
    return out.sort_values(["channel", "t_start"], ignore_index=True)


def _t(times, interval, field):
    if times is None:
        return float(interval) + (1.0 if field == "t_end" else 0.0)
    return float(times.at[interval, field])


def _segment(ch, run, sub, times):
    idx = [(ch, i) for i in run]
    rs = {r for v in sub.loc[idx, "reasons"] for r in str(v).split("+") if r}
    t0, t1 = _t(times, run[0], "t_start"), _t(times, run[-1], "t_end")
    return dict(
        channel=ch,
        t_start=t0,
        t_end=t1,
        duration=t1 - t0,
        severity="bad",
        reasons="+".join(sorted(rs)),
        n_intervals=len(run),
    )


#: Thresholds for whole-recording metrics, ported from the reference EEG
#: analysis. Use with ``IntervalGrid.whole``; they do **not** transfer to short
#: windows (see :data:`DEFAULT_FILTERS`).
#:
#: Deliberately absent from both sets: a bridge filter on ``max_corr``. On a
#: common-reference recording the shared reference and drift push nearly every
#: pairwise correlation above 0.97, so such a filter flags most of the montage
#: while still missing real bridges — see
#: :class:`~signal_quality.metrics.spatial.MaxCorrelation`. Use
#: :func:`~signal_quality.metrics.spatial.correlation_pairs` instead.
WHOLE_RECORDING_FILTERS = [
    Threshold(metric="flat_frac", op=">", value=0.02, flag="FLAT", severity="bad"),
    Threshold(metric="line_ratio", op=">", value=300, flag="LINE_NOISE", severity="bad"),
    Threshold(metric="line_ratio", op=">", value=100, flag="LINE_NOISE", severity="marginal"),
    RobustZ(metric="rms", abs_gt=3.5, flag="AMP_OUTLIER", severity="bad"),
    RobustZ(metric="rms", abs_gt=2.5, flag="AMP_OUTLIER", severity="marginal"),
    Threshold(metric="max_corr", op="<", value=0.6, flag="ISOLATED", severity="bad"),
    Threshold(metric="clip_pct", op=">", value=0.005, flag="CLIPPING", severity="bad"),
    Threshold(metric="clip_pct", op=">", value=0.0, flag="CLIPPING", severity="marginal"),
    Threshold(metric="emg_pct", op=">", value=15, flag="EMG", severity="marginal"),
]

#: Defaults for the 1-second grid. These are **not** the whole-recording
#: thresholds: a one-second estimate is far noisier, and several metrics change
#: meaning at that scale. Calibrated against the synthetic ground truth and the
#: reference study; re-tune per modality rather than assuming they transfer.
DEFAULT_FILTERS = [
    # Half of a one-second window being flat is already a dead channel; the
    # whole-recording 2% cut-off is meaningless when there are only two
    # half-second sub-windows to average over.
    Threshold(metric="flat_frac", op=">", value=0.5, flag="FLAT", severity="bad"),
    # Calibrated on the reference study, where per-channel median line_ratio
    # over 4 s windows separates cleanly: known-bad electrodes 11,250-11,585,
    # every clean electrode <= 899. The whole-recording thresholds (300/100)
    # would flag 87% of all cells here, because a single 4 s window is a far
    # less smoothed spectral estimate than one averaged over half an hour.
    Threshold(metric="line_ratio", op=">", value=3000, flag="LINE_NOISE", severity="bad"),
    Threshold(metric="line_ratio", op=">", value=1000, flag="LINE_NOISE", severity="marginal"),
    RobustZ(metric="rms", abs_gt=4.0, flag="AMP_OUTLIER", severity="bad"),
    RobustZ(metric="rms", abs_gt=3.0, flag="AMP_OUTLIER", severity="marginal"),
    # A single second carries much less of the shared low-frequency drift that
    # couples channels across a whole recording, so correlations run lower and
    # the whole-recording 0.6 cut-off would flag ordinary channels.
    Threshold(metric="max_corr", op="<", value=0.35, flag="ISOLATED", severity="bad"),
    # Within one second, any sample at the converter rail is worth flagging.
    Threshold(metric="clip_pct", op=">", value=0.0, flag="CLIPPING", severity="bad"),
    Threshold(metric="emg_pct", op=">", value=35, flag="EMG", severity="marginal"),
    # Per-interval peak-to-peak only becomes meaningful at this granularity:
    # over a whole recording one movement artifact would set it for everything.
    Threshold(metric="p2p", op=">", value=300, flag="ARTIFACT", severity="bad"),
]
