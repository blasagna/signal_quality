"""The joined metric table."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MetricFrame:
    """Metrics for every (channel, interval), one column per metric.

    ``table`` is indexed by ``(channel, interval)`` and carries ``t_start``,
    ``t_end``, ``coverage``, then one column per metric::

                          t_start  t_end  coverage    rms  line_ratio  emg_pct
        channel interval
        C3      0             0.0   60.0       1.0    4.1        22.0      3.2
        F9      0             0.0   60.0       1.0   41.2       912.0      8.1

    ``notes`` records metrics that could not be computed (e.g. clipping on a
    source without raw counts) so an all-NaN column is never mistaken for a
    clean result.
    """

    table: pd.DataFrame
    grid: object = None
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        n_ch = self.table.index.get_level_values("channel").nunique()
        n_iv = self.table.index.get_level_values("interval").nunique()
        return (
            f"<MetricFrame {n_ch} channels x {n_iv} intervals, metrics: {', '.join(self.metrics)}>"
        )

    @property
    def metrics(self) -> list[str]:
        """Metric column names, excluding the interval bookkeeping columns."""
        return [c for c in self.table.columns if c not in ("t_start", "t_end", "coverage")]

    def long(self) -> pd.DataFrame:
        """Tidy view: one row per (channel, interval, metric, value).

        Convenient for faceted plotting and for comparing metrics on one axis.
        """
        return (
            self.table[self.metrics]
            .stack(future_stack=True)
            .rename("value")
            .reset_index()
            .rename(columns={"level_2": "metric"})
        )

    def per_channel(self, agg="median") -> pd.DataFrame:
        """Collapse the interval axis, giving one row per channel."""
        return self.table[self.metrics].groupby("channel").agg(agg)

    def interval_times(self) -> pd.DataFrame:
        """The interval bookkeeping columns, deduplicated to one row each."""
        return (
            self.table[["t_start", "t_end", "coverage"]]
            .droplevel("channel")
            .groupby(level="interval")
            .first()
        )
