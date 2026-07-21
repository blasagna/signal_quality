"""Raw-signal snippets, so every flag can be checked by eye.

An automated flag is a hypothesis. These plots are how you falsify it — a
"flat" channel that turns out to have a visible trace at low gain, or a
"clipping" flag driven by three isolated samples, are both things only a look at
the signal will reveal.
"""
from __future__ import annotations

import numpy as np

from ..core.context import MetricContext


def plot_channel_snippet(rec, channels, t_start: float = 0.0,
                         duration: float = 10.0, band=(1.0, 45.0), ax=None,
                         step: float = 150.0, title=None):
    """Stacked traces for a few channels over a short window."""
    import matplotlib.pyplot as plt

    names = [channels] if isinstance(channels, str) else list(channels)
    ctx = MetricContext(rec, None)
    idx = {c: i for i, c in enumerate(ctx.ch_names)}
    missing = [c for c in names if c not in idx]
    if missing:
        raise KeyError(f"unknown channel(s): {missing}")

    i0 = max(0, int(t_start * rec.sfreq))
    i1 = min(rec.n_times, int((t_start + duration) * rec.sfreq))
    X = ctx.signal if band is None else ctx.filtered(*band)
    seg = X[[idx[c] for c in names], i0:i1]
    t = np.arange(seg.shape[1]) / rec.sfreq + t_start

    if ax is None:
        _, ax = plt.subplots(figsize=(13, max(2.5, 0.6 * len(names) + 1.5)))
    for k, name in enumerate(names[::-1]):
        ax.plot(t, seg[len(names) - 1 - k] + k * step, lw=0.6)
    ax.set_yticks([k * step for k in range(len(names))])
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 1)
    ax.set_title(title or f"Signal t={t_start:.0f}–{t_start + duration:.0f}s "
                          f"({step:g} µV/div)")
    ax.margins(y=0.02)
    return ax


def plot_flagged_examples(rec, mf, flags, flag: str, metric: str,
                          n: int = 3, duration: float = 10.0, band=(1.0, 45.0)):
    """Worst flagged channels next to the best unflagged ones, same time window.

    Side-by-side at identical gain is the honest comparison: it shows whether
    the flagged channel actually looks different, rather than asking you to
    trust the number.
    """
    import matplotlib.pyplot as plt

    table = getattr(mf, "table", mf)
    per_ch = table[metric].groupby(level="channel").median().dropna()
    bad_set = set(flags[flags["flag"] == flag]["channel"]) if len(flags) else set()
    if not bad_set:
        raise ValueError(f"no channels flagged {flag!r}")

    bad = per_ch[per_ch.index.isin(bad_set)].nlargest(n).index.tolist()
    good = per_ch[~per_ch.index.isin(bad_set)].nsmallest(n).index.tolist()

    t0 = _quiet_window(rec, duration)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    plot_channel_snippet(rec, bad, t0, duration, band=band, ax=axes[0],
                         title=f"Flagged {flag} — worst {len(bad)} by {metric}")
    plot_channel_snippet(rec, good, t0, duration, band=band, ax=axes[1],
                         title=f"Unflagged — best {len(good)} by {metric}")
    fig.tight_layout()
    return fig


def _quiet_window(rec, duration: float) -> float:
    """Pick a start time whose window is fully covered by recorded data."""
    n = int(duration * rec.sfreq)
    cov = rec.covered
    if n >= len(cov):
        return 0.0
    # First fully-covered window, scanning at 1 s resolution.
    stride = max(1, int(rec.sfreq))
    for i in range(0, len(cov) - n, stride):
        if cov[i:i + n].all():
            return i / rec.sfreq
    return 0.0
