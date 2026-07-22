"""Time-axis views: what data exists, and when quality changed."""

from __future__ import annotations

import numpy as np

from ._scale import label_with_range


def plot_availability(rec, issues=None, ax=None, title=None):
    """Green where data exists, red where it does not.

    Timestamp anomalies from the integrity checks are marked underneath, so a
    gap that coincides with a clock fault is visible as such rather than being
    read as two unrelated problems.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 1.9))

    total = rec.duration
    ax.broken_barh([(0, total)], (0.35, 0.65), facecolors="#2e7d32")

    cov = rec.covered
    edges = np.diff(cov.astype(np.int8))
    starts = np.where(edges == -1)[0] + 1
    for st in starts:
        nxt = np.where(edges[st:] == 1)[0]
        end = st + (nxt[0] + 1 if len(nxt) else len(cov) - st)
        ax.broken_barh(
            [(st / rec.sfreq, (end - st) / rec.sfreq)], (0.35, 0.65), facecolors="#c62828"
        )

    if issues is not None and len(issues):
        clock = issues[
            issues["check"].isin(
                [
                    "nonmonotonic_time",
                    "irregular_sampling",
                    "overlapping_packets",
                    "segment_inconsistent",
                ]
            )
        ]
        for _, r in clock.iterrows():
            ax.broken_barh(
                [(r["t_start"], max(r["t_end"] - r["t_start"], total * 0.002))],
                (0.05, 0.22),
                facecolors="#f9a825",
            )
        if len(clock):
            ax.text(0, 0.16, " clock ", va="center", ha="right", fontsize=7)

    ax.set_xlim(0, total)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    missing = 100 * (1 - cov.mean())
    ax.set_title(
        title
        or f"Data availability — {missing:.1f}% missing "
        "(green = data, red = gap, amber = clock anomaly)"
    )
    return ax


def _mask_uncovered(series, table, min_coverage: float = 0.5):
    """Blank out intervals with too little recorded data to summarise.

    Plotting a number for an interval that sits inside a gap invents a
    measurement where none exists.
    """
    if "coverage" not in table.columns:
        return series
    cov = table.groupby(level="interval")["coverage"].first()
    return series.where(cov.reindex(series.index) >= min_coverage)


def plot_metric_trend(
    mf, metric: str, rec=None, ax=None, agg="median", threshold=None, title=None, logy="auto"
):
    """A metric over time, aggregated across channels per interval.

    Only meaningful on a windowed grid — with ``IntervalGrid.whole`` there is a
    single point. Recording gaps are shaded so a dip in quality is not confused
    with absent data, and intervals with no coverage are left blank rather than
    plotted as a value.

    ``logy="auto"`` switches to a log axis when the data spans orders of
    magnitude or sits far below ``threshold`` — otherwise a threshold like 300
    against a median of 0.4 flattens every real datum onto the axis.
    """
    import matplotlib.pyplot as plt

    table = getattr(mf, "table", mf)
    times = table.groupby(level="interval")[["t_start", "t_end"]].first()
    series = table.groupby(level="interval")[metric].agg(agg)
    series = _mask_uncovered(series, table)
    mid = (times["t_start"] + times["t_end"]) / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(mid / 60, series.to_numpy(), "o-", ms=3, lw=1)

    v = series.to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if logy == "auto":
        span = (v.max() / v.min()) if v.size and v.min() > 0 else 1.0
        far = threshold is not None and v.size and threshold > 10 * v.max()
        logy = bool(span > 100 or far)
    if logy and v.size:
        ax.set_yscale("log")

    if threshold is not None:
        ax.axhline(threshold, color="crimson", ls="--", lw=0.9, label=f"threshold {threshold:g}")
        ax.legend(fontsize=8)

    if rec is not None:
        cov = rec.covered
        edges = np.diff(cov.astype(np.int8))
        for st in np.where(edges == -1)[0] + 1:
            nxt = np.where(edges[st:] == 1)[0]
            end = st + (nxt[0] + 1 if len(nxt) else len(cov) - st)
            ax.axvspan(st / rec.sfreq / 60, end / rec.sfreq / 60, color="grey", alpha=0.3)

    ax.set_xlabel("time (min)")
    label_with_range(ax, series.to_numpy(), f"{agg} {metric}")
    ax.set_title(title or f"{metric} over time ({agg} across channels; grey = recording gap)")
    return ax


def plot_clean_fraction(mf, rec=None, metric: str = "p2p", threshold: float = 150.0, ax=None):
    """Fraction of channels under an artifact threshold, per interval.

    The movement-artifact trace: a recording can be perfectly continuous and
    still be unusable for stretches.
    """
    import matplotlib.pyplot as plt

    table = getattr(mf, "table", mf)
    times = table.groupby(level="interval")[["t_start", "t_end"]].first()
    ok = table[metric] <= threshold
    frac = ok.where(table[metric].notna()).groupby(level="interval").mean() * 100
    # An interval inside a recording gap has no data to be clean or dirty. Left
    # as a number it would read as 0% clean, i.e. total artifact, which is the
    # opposite of "nothing was recorded here".
    frac = _mask_uncovered(frac, table)
    mid = (times["t_start"] + times["t_end"]) / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(mid / 60, frac.to_numpy(), "o-", ms=3, lw=1)
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("time (min)")
    label_with_range(ax, frac.to_numpy(), f"% channels {metric} <= {threshold:g}", unit="%")
    ax.set_title(f"Artifact-free fraction over time ({metric} <= {threshold:g} µV)")

    if rec is not None:
        cov = rec.covered
        edges = np.diff(cov.astype(np.int8))
        for st in np.where(edges == -1)[0] + 1:
            nxt = np.where(edges[st:] == 1)[0]
            end = st + (nxt[0] + 1 if len(nxt) else len(cov) - st)
            ax.axvspan(st / rec.sfreq / 60, end / rec.sfreq / 60, color="grey", alpha=0.3)
    return ax
