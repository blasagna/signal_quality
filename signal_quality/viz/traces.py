"""Raw-signal snippets, so every flag can be checked by eye.

An automated flag is a hypothesis. These plots are how you falsify it — a
"flat" channel that turns out to have a visible trace at low gain, or a
"clipping" flag driven by three isolated samples, are both things only a look at
the signal will reveal.
"""
from __future__ import annotations

import numpy as np

from ..core.context import MetricContext
from ._scale import add_scale_bar


def plot_overview(rec, ch_type=None, band=(0.5, 40.0), ax=None, step=None,
                  max_points: int = 1500, clip: bool = True, title=None,
                  normalize="auto"):
    """Every channel over the whole recording — the first thing to look at.

    Metrics summarise; this is the unsummarised signal, and it catches things no
    threshold was written for. Three details matter for honesty:

    * **Envelopes, not samples.** A long recording has far more samples than the
      figure has pixels, so each pixel column shows the min–max range of the
      samples it covers. Naive decimation would drop the brief excursions that
      are usually the most diagnostic thing on the plot.
    * **Traces are clipped to their lane** by default. One channel swinging to
      the converter rail is thousands of times larger than the EEG and would
      otherwise flatten every other channel into a line. Persistently off-scale
      channels are named in the caption rather than silently squashed.
    * **Mixed channel types are normalised per channel.** EEG in microvolts and
      a DC or position channel in device units cannot share one absolute scale —
      on a real study the DC channels set the spacing and every EEG trace
      collapses to a hairline. With ``normalize`` the lanes are in each
      channel's own standard deviations, so all of them are inspectable; the
      cost is that amplitudes are no longer comparable *between* channels, so
      the scale bar reads in SD rather than µV. ``"auto"`` normalises only when
      more than one channel type is present.
    """
    import matplotlib.pyplot as plt

    ctx = MetricContext(rec, None, ch_type=ch_type)
    X = ctx.filtered(*band) if band else ctx.signal
    names = ctx.ch_names
    n_ch, n = X.shape

    types = [str(t) for t in ctx.rec.ds.coords["ch_type"].values]
    if normalize == "auto":
        normalize = len(set(types)) > 1
    if normalize:
        cov0 = ctx.covered
        sd = np.std(X[:, cov0] if cov0.any() else X, axis=1)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        X = X / sd[:, None]

    # Per-column min/max envelope.
    n_bins = int(min(max_points, n))
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    lo = np.empty((n_ch, n_bins))
    hi = np.empty((n_ch, n_bins))
    for k in range(n_bins):
        a, b = edges[k], max(edges[k + 1], edges[k] + 1)
        seg = X[:, a:b]
        lo[:, k], hi[:, k] = seg.min(axis=1), seg.max(axis=1)
    t = (edges[:-1] + np.diff(edges) / 2) / rec.sfreq / 60.0

    # Lane spacing from a robust amplitude, so outliers do not set the scale.
    cov = ctx.covered
    spread = np.std(X[:, cov] if cov.any() else X, axis=1)
    if step is None:
        step = 6.0 * float(np.median(spread[np.isfinite(spread)]) or 1.0)

    # Neighbouring traces overlapping a little is normal in a stacked EEG page,
    # so allow a full lane of headroom before clipping.
    #
    # Report a channel only when it is *persistently* off-scale, judged by how
    # much of its length gets clipped. Keying on peak amplitude instead would
    # name almost every channel on a real recording, where isolated electrode
    # pops are ubiquitous — and a channel clipped only at a few transients is
    # not being misrepresented by the plot.
    unit = "SD" if normalize else "µV"
    clipped = []
    if clip:
        limit = step
        over = (hi > limit) | (lo < -limit)
        frac = over.mean(axis=1)
        for i in np.argsort(-frac):
            if frac[i] <= 0.02:
                break
            peak = max(abs(float(hi[i].max())), abs(float(lo[i].min())))
            clipped.append(f"{names[i]} ({100 * frac[i]:.0f}% off-scale, "
                           f"peak {peak:,.0f} {unit})")
        lo = np.clip(lo, -limit, limit)
        hi = np.clip(hi, -limit, limit)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(4, 0.28 * n_ch + 1.5)))

    for i, name in enumerate(names):
        y = (n_ch - 1 - i) * step
        ax.fill_between(t, lo[i] + y, hi[i] + y, lw=0, color="#1f4e79")

    # Shade recording gaps so absent data is not read as a quiet signal.
    edges_c = np.diff(cov.astype(np.int8))
    for s in np.where(edges_c == -1)[0] + 1:
        nxt = np.where(edges_c[s:] == 1)[0]
        e = s + (nxt[0] + 1 if len(nxt) else len(cov) - s)
        ax.axvspan(s / rec.sfreq / 60, e / rec.sfreq / 60, color="#c62828",
                   alpha=0.18, zorder=0)

    ax.set_yticks([(n_ch - 1 - i) * step for i in range(n_ch)])
    ax.set_yticklabels(names, fontsize=7)
    ax.set_ylim(-step, n_ch * step)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (min)")
    ax.set_ylabel(f"channel  (lane spacing {step:,.0f} {unit})")

    band_txt = f"{band[0]:g}–{band[1]:g} Hz" if band else "wideband"
    scale_txt = "per-channel SD" if normalize else "shared µV scale"
    ax.set_title(title or f"All {n_ch} channels, whole recording "
                          f"({band_txt}; {scale_txt}; red = gap)")
    add_scale_bar(ax, step, unit)
    if clipped:
        shown, extra = clipped[:4], len(clipped) - 4
        note = "clipped to lane for display: " + ", ".join(shown)
        if extra > 0:
            note += f", and {extra} more"
        ax.text(0.0, -0.16 if n_ch < 20 else -0.09, note,
                transform=ax.transAxes, fontsize=7, style="italic", color="#c62828")
    return ax


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
    # Peak excursion is the number a reader actually wants off a trace plot, and
    # the channel-name ticks cannot convey it.
    peak = float(np.abs(seg).max()) if seg.size else 0.0
    ax.set_ylabel(f"channel  (lane spacing {step:g} µV)")
    ax.set_title(title or f"Signal t={t_start:.0f}–{t_start + duration:.0f}s "
                          f"({step:g} µV/div; peak |x| = {peak:,.0f} µV)")
    ax.margins(y=0.02)
    add_scale_bar(ax, step, "µV")
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
