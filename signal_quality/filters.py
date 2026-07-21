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

FLAG_COLUMNS = ["channel", "interval", "flag", "severity", "metric", "value",
                "threshold"]

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
        return pd.DataFrame({
            "channel": idx.get_level_values("channel"),
            "interval": idx.get_level_values("interval"),
            "flag": self.flag,
            "severity": self.severity,
            "metric": metric,
            "value": np.asarray(values)[mask.to_numpy()],
            "threshold": threshold,
        })


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
        thr = self.abs_gt if self.abs_gt is not None else (
            self.gt if self.gt is not None else self.lt)
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
        ["channel", "interval", "severity"], ignore_index=True)


def verdict(flags: pd.DataFrame, channels=None) -> pd.DataFrame:
    """Collapse flags to one good/marginal/bad verdict per channel.

    Single implementation of the categorisation that the reference project kept
    in two divergent copies. ``reasons`` lists the distinct flags that fired, in
    severity order, for labelling plots.
    """
    order = {"good": 0, "marginal": 1, "bad": 2}
    names = list(channels) if channels is not None else sorted(
        flags["channel"].unique()) if len(flags) else []

    rows = []
    for ch in names:
        f = flags[flags["channel"] == ch] if len(flags) else flags
        if not len(f):
            rows.append(dict(channel=ch, verdict="good", reasons="", n_flags=0))
            continue
        worst = max(f["severity"], key=lambda s: order.get(s, 0))
        top = f[f["severity"] == worst]["flag"].unique()
        rows.append(dict(channel=ch, verdict=worst, reasons="+".join(sorted(top)),
                         n_flags=len(f)))
    out = pd.DataFrame(rows, columns=["channel", "verdict", "reasons", "n_flags"])
    return out.set_index("channel") if len(out) else out


#: Thresholds ported from the reference EEG analysis. Reasonable defaults for
#: clinical scalp EEG; re-tune per modality rather than assuming they transfer.
#:
#: Deliberately absent: a bridge filter on ``max_corr``. On a common-reference
#: recording the shared reference and drift push nearly every pairwise
#: correlation above 0.97, so such a filter flags most of the montage while
#: still missing real bridges — see
#: :class:`~signal_quality.metrics.spatial.MaxCorrelation`. Use
#: :func:`~signal_quality.metrics.spatial.correlation_pairs` to shortlist
#: candidates instead.
DEFAULT_FILTERS = [
    Threshold(metric="flat_frac", op=">", value=0.02,
              flag="FLAT", severity="bad"),
    Threshold(metric="line_ratio", op=">", value=300,
              flag="LINE_NOISE", severity="bad"),
    Threshold(metric="line_ratio", op=">", value=100,
              flag="LINE_NOISE", severity="marginal"),
    RobustZ(metric="rms", abs_gt=3.5, flag="AMP_OUTLIER", severity="bad"),
    RobustZ(metric="rms", abs_gt=2.5, flag="AMP_OUTLIER", severity="marginal"),
    Threshold(metric="max_corr", op="<", value=0.6,
              flag="ISOLATED", severity="bad"),
    Threshold(metric="clip_pct", op=">", value=0.005,
              flag="CLIPPING", severity="bad"),
    # Any clipping at all is worth surfacing: hitting the converter rail even
    # briefly means the amplifier range was wrong for that channel.
    Threshold(metric="clip_pct", op=">", value=0.0,
              flag="CLIPPING", severity="marginal"),
    Threshold(metric="emg_pct", op=">", value=15,
              flag="EMG", severity="marginal"),
]
