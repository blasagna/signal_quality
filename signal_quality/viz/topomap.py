"""Electrode positions on the head model.

Ported from the reference project's contact-quality figure, generalised to any
metric column rather than hard-coding mains pickup.
"""
from __future__ import annotations

import numpy as np

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


def plot_metric_topomap(mf, metric: str, ax=None, log: bool = False,
                        cmap: str = "RdYlGn_r", interval=None, title=None):
    """Spatially interpolated map of one metric across the scalp.

    The interpolation between electrodes is illustrative — only the electrode
    locations carry measurements — so pair this with
    :func:`plot_verdict_topomap`, which shows the per-electrode calls without
    smoothing.
    """
    import matplotlib.pyplot as plt
    import mne

    values = _per_channel(mf, metric, interval)
    placed, xy, unplaced = place(values.index)
    if not placed:
        raise ValueError("no channels could be given a scalp position")

    v = values.loc[placed].to_numpy(dtype=float)
    if log:
        v = np.log10(np.clip(v, 1e-12, None))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(v, xy, axes=ax, show=False, cmap=cmap,
                                 names=placed, contours=4)
    ax.set_title(title or f"{metric}{' (log10)' if log else ''}", fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.65)
    cb.set_label(f"log10({metric})" if log else metric)
    if unplaced:
        ax.text(0.5, -0.08, "no standard position: " + ", ".join(unplaced),
                transform=ax.transAxes, ha="center", fontsize=7, style="italic")
    return ax


def plot_verdict_topomap(verdicts, ax=None, title="Per-electrode verdict"):
    """Categorical good/marginal/bad per electrode, with the reason for each.

    This is the authoritative panel: one marker per electrode, no interpolation,
    labelled with the flags that fired.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import mne

    placed, xy, unplaced = place(verdicts.index)
    if not placed:
        raise ValueError("no channels could be given a scalp position")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    # Blank head outline to draw the markers onto.
    mne.viz.plot_topomap(np.zeros(len(placed)), xy, axes=ax, show=False,
                         cmap="Greys", vlim=(0, 1), contours=0, sensors=False)

    for k, ch in enumerate(placed):
        row = verdicts.loc[ch]
        ax.scatter(*xy[k], s=430, c=VERDICT_COLORS.get(row["verdict"], "#999"),
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.text(xy[k][0], xy[k][1], ch, ha="center", va="center", fontsize=7,
                color="w", zorder=6, fontweight="bold")
        if row["reasons"]:
            # Push the reason outward from centre so peripheral labels do not
            # collide with the markers.
            r = np.hypot(*xy[k]) + 1e-9
            dx, dy = xy[k] / r * 0.017
            ax.text(xy[k][0] + dx, xy[k][1] + dy - 0.004, _short(row["reasons"]),
                    ha="center", va="top", fontsize=5.5, color="#333", zorder=6)

    counts = verdicts.loc[placed, "verdict"].value_counts()
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=VERDICT_COLORS[g], mec="k", ms=11,
               label=f"{g} ({int(counts.get(g, 0))})")
        for g in ("good", "marginal", "bad")],
        loc="lower right", fontsize=9, frameon=True)
    # Extra pad: reason labels sit above the topmost electrodes and would
    # otherwise run into the title.
    ax.set_title(title, fontsize=12, pad=18)
    ax.margins(0.12)
    if unplaced:
        ax.text(0.5, -0.06, "no standard position: " + ", ".join(unplaced),
                transform=ax.transAxes, ha="center", fontsize=7, style="italic")
    return ax


def plot_contact_quality(mf, verdicts, metric: str = "line_ratio",
                         log: bool = True, suptitle=None):
    """The two-panel contact-quality figure: interpolated map + verdicts."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    plot_metric_topomap(mf, metric, ax=axes[0], log=log,
                        title=f"{metric} — higher = worse contact\n"
                              "(interpolated; per-electrode verdicts at right)")
    plot_verdict_topomap(verdicts, ax=axes[1])
    fig.suptitle(suptitle or "Electrode contact quality — derived from the signal",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def _per_channel(mf, metric, interval=None):
    table = getattr(mf, "table", mf)
    if metric not in table.columns:
        raise KeyError(f"{metric!r} not in the metric table; have "
                       f"{list(table.columns)}")
    if interval is not None:
        return table.xs(interval, level="interval")[metric]
    return table[metric].groupby(level="channel").median()
