"""The metric protocol.

A metric is one quality check. It declares what it needs, what columns it
produces, and computes those columns over a grid of intervals. It knows nothing
about thresholds — deciding whether a value is *bad* is the filters' job, so the
same metric serves a strict and a lenient policy without modification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Metric:
    """Base class for a per-channel metric.

    Subclasses implement :meth:`compute_interval`, returning one value per
    channel for a single interval. Requirements listed in ``requires`` that the
    recording cannot satisfy (e.g. ``counts`` for a source that only stores
    volts) yield NaN columns and a recorded reason, rather than an exception —
    a partially-assessable recording is still worth assessing.
    """

    name: str = "metric"
    requires: tuple[str, ...] = ()
    unavailable_reason: str | None = field(default=None, init=False, repr=False)

    @property
    def columns(self) -> list[str]:
        return [self.name]

    def analysis_band(self):
        """Band this metric filters to, or None for wideband.

        Used to size block padding, so chunked processing reproduces
        whole-recording filtering exactly. Subclasses whose band is not simply
        ``(l_freq, h_freq)`` override this.
        """
        lo, hi = getattr(self, "l_freq", None), getattr(self, "h_freq", None)
        return None if lo is None and hi is None else (lo, hi)

    def required_pad_s(self, sfreq: float) -> float:
        """Seconds of context this metric needs either side of an interval.

        The two contributions **add**. A widened analysis window reaches
        ``min_analysis_s / 2`` beyond the interval, and every filtered sample out
        there needs its own filter half-length of clean data beyond *that*.
        Taking the maximum instead leaves the outermost analysis samples filtered
        against block edge, which showed up as ``emg_pct`` differing from
        whole-recording values on 29% of intervals.
        """
        from .blocks import filter_pad_samples

        pad = 0.0
        b = self.analysis_band()
        if b is not None:
            pad = filter_pad_samples(sfreq, b[0], b[1]) / sfreq
        return pad + float(getattr(self, "min_analysis_s", 0.0)) / 2

    def compute_interval(self, ctx, i_start: int, i_stop: int) -> np.ndarray:
        raise NotImplementedError

    def available(self, ctx) -> list[str]:
        """Requirements this recording cannot satisfy."""
        return [r for r in self.requires if not self._available(ctx, r)]

    def compute(self, ctx, grid) -> pd.DataFrame:
        """Run over every interval -> DataFrame indexed by (channel, interval).

        Kept for single-metric use; :func:`compute` below is the block-aware
        path that several metrics should share.
        """
        return compute([self], ctx.rec, grid, ch_type=None, _ctx=ctx).table[
            self.columns]

    @staticmethod
    def _available(ctx, req: str) -> bool:
        if req == "counts":
            return ctx.counts is not None
        return True


def compute(metrics, rec, grid=None, ch_type: str | None = "eeg",
            block_s: float | None = None, _ctx=None):
    """Compute several metrics over one grid and join them into one table.

    Because every metric is handed the same ``grid``, the returned frames share
    a row index exactly, so the join introduces no NaN padding and no reordering.

    Work proceeds **block by block**: each block loads a padded span, filters it
    once, computes every metric for the intervals inside it, then releases the
    buffers. Padding is sized from the metrics' own requirements so the result
    is numerically identical to processing the whole recording at once.

    ``grid`` defaults to 1-second non-overlapping windows.

    Returns a :class:`~signal_quality.core.frame.MetricFrame`.
    """
    from .blocks import DEFAULT_BLOCK_S, filter_pad_samples, plan_blocks
    from .context import MetricContext
    from .frame import MetricFrame
    from .intervals import IntervalGrid

    if grid is None:
        grid = IntervalGrid.fixed(rec, IntervalGrid.DEFAULT_WINDOW)

    ctx = _ctx if _ctx is not None else MetricContext(rec, grid, ch_type=ch_type)
    metrics = list(metrics)

    notes, missing = [], {}
    for m in metrics:
        miss = m.available(ctx)
        if miss:
            missing[id(m)] = miss
            m.unavailable_reason = (
                f"{m.name}: requires {', '.join(miss)}, not available from this "
                f"source")
            notes.append(m.unavailable_reason)

    pad_s = max([m.required_pad_s(ctx.sfreq) for m in metrics] or [0.0])
    pad = int(np.ceil(pad_s * ctx.sfreq))
    # A view shorter than the filter itself would distort regardless of padding.
    min_view = 2 * max(
        [filter_pad_samples(ctx.sfreq, *m.analysis_band()) for m in metrics
         if m.analysis_band() is not None] or [0])

    blocks = plan_blocks(grid, ctx.n_times, pad,
                         block_s if block_s is not None else DEFAULT_BLOCK_S,
                         sfreq=ctx.sfreq, min_view=min_view)

    # Accumulate each metric's per-interval output as plain (n_ch, n_cols)
    # ndarrays and assemble one DataFrame per metric at the end. Building a
    # DataFrame with a fresh MultiIndex per (interval, metric) instead turned
    # the join into thousands of tiny-object constructions on a 1-second grid.
    n_ch = len(ctx.ch_names)
    by_metric = {id(m): [] for m in metrics}
    iid_order = []
    bounds = grid.table[["i_start", "i_stop"]]
    for blk in blocks:
        ctx.set_view(blk.view_start, blk.view_stop)
        for iid in blk.interval_ids:
            i0, i1 = int(bounds.at[iid, "i_start"]), int(bounds.at[iid, "i_stop"])
            iid_order.append(iid)
            for m in metrics:
                if id(m) in missing:
                    by_metric[id(m)].append(np.full((n_ch, len(m.columns)), np.nan))
                    continue
                vals = np.asarray(m.compute_interval(ctx, i0, i1), dtype=float)
                by_metric[id(m)].append(vals.reshape(n_ch, -1))
        ctx.release()
    ctx.set_view(0, ctx.n_times)

    n_iv = len(iid_order)
    index = pd.MultiIndex.from_arrays(
        [np.tile(ctx.ch_names, n_iv), np.repeat(iid_order, n_ch)],
        names=["channel", "interval"])
    frames = [pd.DataFrame(np.concatenate(by_metric[id(m)], axis=0),
                           columns=m.columns, index=index)
              for m in metrics]
    table = pd.concat(frames, axis=1).sort_index()
    meta = grid.table.reindex(
        table.index.get_level_values("interval")).set_index(table.index)
    table = pd.concat([meta[["t_start", "t_end", "coverage"]], table], axis=1)
    return MetricFrame(table, grid=grid, notes=notes)
