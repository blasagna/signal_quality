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

    def compute_interval(self, ctx, i_start: int, i_stop: int) -> np.ndarray:
        raise NotImplementedError

    def compute(self, ctx, grid) -> pd.DataFrame:
        """Run over every interval -> DataFrame indexed by (channel, interval)."""
        missing = [r for r in self.requires if not self._available(ctx, r)]
        n_ch = len(ctx.ch_names)
        blocks = []
        for iid, i0, i1 in grid:
            if missing:
                vals = np.full((n_ch, len(self.columns)), np.nan)
            else:
                vals = np.asarray(self.compute_interval(ctx, i0, i1), dtype=float)
                vals = vals.reshape(n_ch, -1)
            blocks.append(pd.DataFrame(
                vals,
                columns=self.columns,
                index=pd.MultiIndex.from_product(
                    [ctx.ch_names, [iid]], names=["channel", "interval"]),
            ))
        if missing:
            self.unavailable_reason = (
                f"{self.name}: requires {', '.join(missing)}, not available from "
                f"this source")
        return pd.concat(blocks).sort_index()

    @staticmethod
    def _available(ctx, req: str) -> bool:
        if req == "counts":
            return ctx.counts is not None
        if req == "positions":
            return True
        return True


def compute(metrics, rec, grid, ch_type: str | None = "eeg"):
    """Compute several metrics over one grid and join them into one table.

    Because every metric is handed the same ``grid``, the returned frames share
    a row index exactly, so the join introduces no NaN padding and no reordering.

    Returns a :class:`~signal_quality.core.frame.MetricFrame`.
    """
    from .context import MetricContext
    from .frame import MetricFrame

    ctx = MetricContext(rec, grid, ch_type=ch_type)
    frames, notes = [], []
    for m in metrics:
        frames.append(m.compute(ctx, grid))
        if m.unavailable_reason:
            notes.append(m.unavailable_reason)

    table = pd.concat(frames, axis=1)
    meta = grid.table.reindex(
        table.index.get_level_values("interval")).set_index(table.index)
    table = pd.concat([meta[["t_start", "t_end", "coverage"]], table], axis=1)
    return MetricFrame(table, grid=grid, notes=notes)
