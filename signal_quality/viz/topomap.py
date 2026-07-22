"""Electrode positions on the head model.

Ported from the reference project's contact-quality figure, generalised to any
metric column rather than hard-coding mains pickup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..montage import place

VERDICT_COLORS = {"good": "#2e7d32", "marginal": "#f9a825", "bad": "#c62828"}

#: Compact labels for the head plot. Full flag names are several times wider
#: than the electrode spacing and collide into illegibility.
SHORT_FLAGS = {
    "LINE_NOISE": "line",
    "AMP_OUTLIER": "amp",
    "ISOLATED": "isol",
    "CLIPPING": "clip",
    "FLAT": "flat",
    "EMG": "emg",
    "BRIDGED": "bridge",
}


def _short(reasons: str) -> str:
    if not reasons:
        return ""
    return "+".join(SHORT_FLAGS.get(r, r.lower()) for r in reasons.split("+"))


def plot_metric_topomap(
    mf,
    metric: str,
    ax=None,
    log: bool = False,
    cmap: str = "RdYlGn_r",
    interval=None,
    title=None,
    robust: bool = True,
):
    """Spatially interpolated map of one metric across the scalp.

    The interpolation between electrodes is illustrative — only the electrode
    locations carry measurements — so pair this with
    :func:`plot_verdict_topomap`, which shows the per-electrode calls without
    smoothing.

    ``robust`` sets the colour limits from the 2nd–98th percentile rather than
    the full range. Without it a single degenerate channel — a dead electrode
    reads zero, which is minus infinity once logged — stretches the scale so far
    that every other channel renders as one flat colour.
    """
    import matplotlib.pyplot as plt
    import mne

    values = _per_channel(mf, metric, interval)
    placed, xy, unplaced = place(values.index)
    if not placed:
        raise ValueError("no channels could be given a scalp position")

    v = values.loc[placed].to_numpy(dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(np.where(v > 0, v, np.nan))

    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise ValueError(f"{metric!r} has no finite values to plot")
    if robust and finite.size > 3:
        lo, hi = np.percentile(finite, [2, 98])
    else:
        lo, hi = finite.min(), finite.max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    # Degenerate channels are pinned to the bottom of the scale rather than
    # dropped, so their electrode still appears on the map.
    v = np.clip(np.nan_to_num(v, nan=lo, neginf=lo, posinf=hi), lo, hi)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(
        v, xy, axes=ax, show=False, cmap=cmap, names=placed, contours=4, vlim=(lo, hi)
    )
    ax.set_title(title or f"{metric}{' (log10)' if log else ''}", fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.65)
    cb.set_label(f"log10({metric})" if log else metric)
    if unplaced:
        ax.text(
            0.5,
            -0.08,
            "no standard position: " + ", ".join(unplaced),
            transform=ax.transAxes,
            ha="center",
            fontsize=7,
            style="italic",
        )
    return ax


def plot_pct_bad_topomap(summary, ax=None, cmap="RdYlGn_r", vmax=None, title=None):
    """Percentage of covered time each electrode spent flagged bad.

    Takes a :func:`~signal_quality.filters.channel_summary`. This is the
    continuous quantity worth mapping — a binary good/bad label discards the
    difference between an electrode that failed for ten seconds and one that
    never worked.
    """
    import matplotlib.pyplot as plt
    import mne

    placed, xy, unplaced = place(summary.index)
    if not placed:
        raise ValueError("no channels could be given a scalp position")

    v = summary.loc[placed, "pct_bad"].to_numpy(dtype=float)
    v = np.nan_to_num(v, nan=0.0)
    hi = float(vmax if vmax is not None else max(v.max(), 1.0))

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    im, _ = mne.viz.plot_topomap(
        v, xy, axes=ax, show=False, cmap=cmap, names=placed, contours=4, vlim=(0, hi)
    )
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.65)
    cb.set_label("% of covered time flagged bad")
    ax.set_title(title or "Time spent bad, per electrode", fontsize=11, pad=14)
    if unplaced:
        ax.text(
            0.5,
            -0.07,
            "no standard position: " + ", ".join(unplaced),
            transform=ax.transAxes,
            ha="center",
            fontsize=7,
            style="italic",
        )
    return ax


def plot_verdict_topomap(verdicts, ax=None, title="Per-electrode verdict"):
    """Categorical good/marginal/bad per electrode, with the reason for each.

    Takes a :func:`~signal_quality.filters.channel_summary` (one row per
    channel). Per-interval verdicts must be rolled up first — a head plot can
    only show one value per electrode, and which one it should be is a policy
    question that belongs in ``filters``, not here.
    """
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.lines import Line2D

    if isinstance(verdicts.index, pd.MultiIndex):
        raise TypeError(
            "plot_verdict_topomap needs one row per channel; pass "
            "sq.channel_summary(verdicts), or plot_quality_heatmap() to see "
            "the per-interval verdicts over time"
        )

    placed, xy, unplaced = place(verdicts.index)
    if not placed:
        raise ValueError("no channels could be given a scalp position")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    # Blank head outline to draw the markers onto.
    mne.viz.plot_topomap(
        np.zeros(len(placed)),
        xy,
        axes=ax,
        show=False,
        cmap="Greys",
        vlim=(0, 1),
        contours=0,
        sensors=False,
    )

    for k, ch in enumerate(placed):
        row = verdicts.loc[ch]
        ax.scatter(
            *xy[k],
            s=430,
            c=VERDICT_COLORS.get(row["verdict"], "#999"),
            edgecolors="k",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            xy[k][0],
            xy[k][1],
            ch,
            ha="center",
            va="center",
            fontsize=7,
            color="w",
            zorder=6,
            fontweight="bold",
        )
        if row.get("reasons"):
            # Push the reason outward from centre so peripheral labels do not
            # collide with the markers.
            r = np.hypot(*xy[k]) + 1e-9
            dx, dy = xy[k] / r * 0.017
            ax.text(
                xy[k][0] + dx,
                xy[k][1] + dy - 0.004,
                _short(row["reasons"]),
                ha="center",
                va="top",
                fontsize=5.5,
                color="#333",
                zorder=6,
            )

    counts = verdicts.loc[placed, "verdict"].value_counts()
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc=VERDICT_COLORS[g],
                mec="k",
                ms=11,
                label=f"{g} ({int(counts.get(g, 0))})",
            )
            for g in ("good", "marginal", "bad")
        ],
        loc="lower right",
        fontsize=9,
        frameon=True,
    )
    # Extra pad: reason labels sit above the topmost electrodes and would
    # otherwise run into the title.
    ax.set_title(title, fontsize=12, pad=18)
    ax.margins(0.12)
    if unplaced:
        ax.text(
            0.5,
            -0.06,
            "no standard position: " + ", ".join(unplaced),
            transform=ax.transAxes,
            ha="center",
            fontsize=7,
            style="italic",
        )
    return ax


def plot_contact_quality(mf, verdicts, metric: str = "line_ratio", log: bool = True, suptitle=None):
    """The two-panel contact-quality figure: interpolated map + verdicts."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    plot_metric_topomap(
        mf,
        metric,
        ax=axes[0],
        log=log,
        title=f"{metric} — higher = worse contact\n(interpolated; per-electrode verdicts at right)",
    )
    plot_verdict_topomap(verdicts, ax=axes[1])
    fig.suptitle(suptitle or "Electrode contact quality — derived from the signal", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def _per_channel(mf, metric, interval=None):
    table = getattr(mf, "table", mf)
    if metric not in table.columns:
        raise KeyError(f"{metric!r} not in the metric table; have {list(table.columns)}")
    if interval is not None:
        return table.xs(interval, level="interval")[metric]
    return table[metric].groupby(level="channel").median()
