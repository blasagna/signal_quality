"""Channel × time quality map.

This is what per-interval flagging buys, and the plot the whole redesign exists
to make possible. A per-channel verdict can only say *which* electrodes are
suspect; this says **when**, which is what tells an intermittent contact apart
from a dead one, and a subject-wide movement episode apart from an electrode
fault.

``no_data`` is drawn as its own category rather than as a shade of "good".
Missing data is not quality, and colouring a gap green would invite exactly the
wrong conclusion.
"""

from __future__ import annotations

import numpy as np

#: Ordered worst-last; index into this list is what the image actually plots.
VERDICT_LEVELS = ["no_data", "good", "marginal", "bad"]
VERDICT_COLORS = {
    "no_data": "#d9d9d9",
    "good": "#2e7d32",
    "marginal": "#f9a825",
    "bad": "#c62828",
}


def plot_quality_heatmap(verdicts, mf=None, ax=None, channels=None, title=None, time_unit="min"):
    """Verdict for every ``(channel, interval)`` as an image.

    ``mf`` supplies real interval times for the x axis; without it the axis is
    interval index, which is only meaningful for a uniform grid.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    if not len(verdicts):
        raise ValueError("no verdicts to plot")

    grid = verdicts["verdict"].unstack("interval")
    if channels is not None:
        grid = grid.reindex([c for c in channels if c in grid.index])

    codes = np.full(grid.shape, np.nan)
    for k, level in enumerate(VERDICT_LEVELS):
        codes[grid.to_numpy() == level] = k

    t0, t1 = _time_span(mf, grid)
    div = 60.0 if time_unit == "min" else 1.0

    cmap = ListedColormap([VERDICT_COLORS[v] for v in VERDICT_LEVELS])
    norm = BoundaryNorm(np.arange(-0.5, len(VERDICT_LEVELS)), cmap.N)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(3.5, 0.22 * len(grid) + 1.5)))
    ax.imshow(
        codes,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[t0 / div, t1 / div, len(grid) - 0.5, -0.5],
    )

    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=7)
    ax.set_xlabel(f"time ({time_unit})")
    ax.set_ylabel("channel")

    counts = verdicts["verdict"].value_counts()
    total = int(counts.sum())
    ax.legend(
        handles=[
            Patch(
                facecolor=VERDICT_COLORS[v],
                edgecolor="k",
                linewidth=0.4,
                label=f"{v} ({100 * int(counts.get(v, 0)) / total:.1f}%)",
            )
            for v in reversed(VERDICT_LEVELS)
        ],
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        fontsize=8,
        frameon=False,
    )

    win = _window_s(mf, grid)
    ax.set_title(
        title or f"Quality per channel over time ({win:g} s windows, {len(grid.columns)} of them)"
    )
    return ax


def plot_flag_timeline(verdicts, mf=None, ax=None, title=None, time_unit="min"):
    """How many channels carry each flag, over time.

    Separates the two things a heatmap makes you squint for: a fault on one
    electrode, versus an episode that hits the whole head at once (movement,
    a subject touching the leads) and which no per-channel view can express.
    """
    import matplotlib.pyplot as plt

    if not len(verdicts):
        raise ValueError("no verdicts to plot")

    exploded = verdicts[verdicts["reasons"] != ""]["reasons"].str.split("+").explode()
    if not len(exploded):
        raise ValueError("no flags fired; nothing to plot")

    counts = exploded.reset_index().groupby(["interval", "reasons"]).size().unstack(fill_value=0)

    t = _interval_times(mf, counts.index)
    div = 60.0 if time_unit == "min" else 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.8))
    for flag in counts.columns:
        ax.plot(t / div, counts[flag].to_numpy(), lw=1.0, label=flag)

    n_ch = verdicts.index.get_level_values("channel").nunique()
    ax.set_ylim(0, n_ch)
    ax.set_xlabel(f"time ({time_unit})")
    ax.set_ylabel(f"channels flagged\nof {n_ch}")
    ax.legend(fontsize=8, ncol=min(4, len(counts.columns)))
    ax.set_title(
        title
        or "Flags over time — a spike across many channels is an episode, not an electrode fault"
    )
    return ax


def _time_span(mf, grid):
    if mf is None:
        return float(grid.columns.min()), float(grid.columns.max() + 1)
    table = getattr(mf, "table", mf)
    times = table.groupby(level="interval")[["t_start", "t_end"]].first()
    return float(times["t_start"].min()), float(times["t_end"].max())


def _interval_times(mf, intervals):
    if mf is None:
        return np.asarray(intervals, dtype=float)
    table = getattr(mf, "table", mf)
    times = table.groupby(level="interval")[["t_start", "t_end"]].first()
    mid = (times["t_start"] + times["t_end"]) / 2
    return mid.reindex(intervals).to_numpy()


def _window_s(mf, grid):
    if mf is None:
        return 1.0
    table = getattr(mf, "table", mf)
    times = table.groupby(level="interval")[["t_start", "t_end"]].first()
    return float(np.median(times["t_end"] - times["t_start"]))
