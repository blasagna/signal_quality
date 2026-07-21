"""Shared y-axis annotation helpers.

Stacked trace plots label the y axis with channel names, which says nothing
about amplitude — the reader cannot tell 5 µV from 5 mV. These helpers put the
numeric scale back on every plot: a scale bar for stacked axes, and an explicit
observed range on ordinary ones.
"""
from __future__ import annotations

import numpy as np


def add_scale_bar(ax, size: float, unit: str = "µV", loc=(0.985, 0.03),
                  color: str = "k", fontsize: int = 8):
    """Draw a vertical bar spanning ``size`` data units, labelled with its value.

    Used where the y axis carries channel names instead of numbers, so the plot
    still states what an excursion is worth. The label sits to the *left* of the
    bar so it stays inside the axes when the bar is at the right edge.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x], [y, y + size], color=color, lw=2.5, solid_capstyle="butt",
            zorder=10)
    ax.text(x - 0.008 * (x1 - x0), y + size / 2, f"{_fmt(size)} {unit}",
            va="center", ha="right", fontsize=fontsize, color=color, zorder=10,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    return ax


def label_with_range(ax, values, label: str, unit: str = "", log: bool = False):
    """Set a y label that states the observed range of the plotted data.

    Reading a value off an axis is guesswork when the interesting variation
    spans orders of magnitude; naming the range makes the plot quantitative
    without needing to squint at ticks.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size:
        suffix = f"\nrange: {_fmt(v.min())} – {_fmt(v.max())}{' ' + unit if unit else ''}"
    else:
        suffix = ""
    ax.set_ylabel(f"{label}{suffix}")
    return ax


def _fmt(x: float) -> str:
    """Compact number formatting that stays readable across many decades."""
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1e5 or ax < 1e-2:
        return f"{x:.1e}"
    if ax >= 100:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"
