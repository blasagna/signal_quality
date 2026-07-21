"""Spectral comparison of flagged against unflagged channels.

The point of this plot is evidentiary: a table saying a channel has a mains
ratio of 912 is an assertion, while a spectrum showing its 60 Hz spike towering
over every clean channel is the evidence. Whenever a flag fires, this is how you
confirm it is real rather than a threshold artefact.
"""
from __future__ import annotations

import numpy as np

from ..core.context import MetricContext


def plot_good_bad_psd(rec, flags, flag: str = "LINE_NOISE", ax=None,
                      fmax: float = 80.0, band=None, max_examples: int = 5,
                      title=None):
    """Overlay PSDs of channels carrying ``flag`` against unflagged channels.

    Flagged channels are drawn worst-first (by the metric value that tripped the
    flag), so the plot shows the strongest evidence rather than whichever
    channel sorts first alphabetically.

    ``band`` shades a frequency range of interest; it defaults to a window
    around the recording's mains frequency for line-noise flags.
    """
    import matplotlib.pyplot as plt

    hit = flags[flags["flag"] == flag] if len(flags) else flags
    if not len(hit):
        raise ValueError(f"no channels flagged {flag!r}")
    # Rank by the most extreme value that fired this flag on each channel.
    ranked = (hit.assign(_v=hit["value"].abs())
              .groupby("channel")["_v"].max()
              .sort_values(ascending=False))
    bad = ranked.index.tolist()

    ctx = MetricContext(rec, None, ch_type="eeg")
    good = [c for c in ctx.ch_names if c not in set(bad)]

    f, P = ctx.welch(0, rec.n_times, band=None)
    idx = {c: i for i, c in enumerate(ctx.ch_names)}
    sel = f <= fmax

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    for c in good:
        if c in idx:
            ax.semilogy(f[sel], P[idx[c]][sel], color="#2e7d32", lw=0.7, alpha=0.4)
    for c in bad[:max_examples]:
        if c in idx:
            ax.semilogy(f[sel], P[idx[c]][sel], lw=1.6,
                        label=f"{c} ({ranked[c]:.0f})")

    if band is None and "LINE" in flag.upper():
        band = (rec.line_freq - 5, rec.line_freq + 5)
    if band:
        ax.axvspan(*band, color="orange", alpha=0.15)

    ax.plot([], [], color="#2e7d32", lw=0.7, alpha=0.6,
            label=f"unflagged ({len(good)})")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (µV²/Hz)")
    ax.set_title(title or f"Spectra: worst {min(max_examples, len(bad))} of "
                          f"{len(bad)} channels flagged {flag}, vs unflagged")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, fmax)
    return ax


def plot_psd_examples(rec, mf, metric: str = "line_ratio", n: int = 3, ax=None):
    """Best and worst ``n`` channels by ``metric``, spectra overlaid.

    Useful when nothing was flagged but you still want to see the spread — the
    ranking is meaningful even where the threshold was not crossed.
    """
    import matplotlib.pyplot as plt

    table = getattr(mf, "table", mf)
    per_ch = table[metric].groupby(level="channel").median().dropna()
    worst = per_ch.nlargest(n).index.tolist()
    best = per_ch.nsmallest(n).index.tolist()

    ctx = MetricContext(rec, None, ch_type="eeg")
    f, P = ctx.welch(0, rec.n_times, band=None)
    idx = {c: i for i, c in enumerate(ctx.ch_names)}
    sel = f <= 80

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    for c in best:
        if c in idx:
            ax.semilogy(f[sel], P[idx[c]][sel], color="#2e7d32", lw=1.2,
                        label=f"{c} (best, {per_ch[c]:.0f})")
    for c in worst:
        if c in idx:
            ax.semilogy(f[sel], P[idx[c]][sel], color="#c62828", lw=1.2,
                        label=f"{c} (worst, {per_ch[c]:.0f})")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (µV²/Hz)")
    ax.set_title(f"Best vs worst channels by {metric}")
    ax.legend(fontsize=8, ncol=2)
    return ax
