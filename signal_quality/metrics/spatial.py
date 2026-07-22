"""Between-channel metrics.

These need the whole channel set at once, so they cannot be expressed as a
threshold on a single-channel measurement — unlike robust-z, which *is* a
cross-channel comparison of an existing column and therefore lives in
``filters.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.metric import Metric


@dataclass
class MaxCorrelation(Metric):
    """Highest absolute correlation with any other channel.

    **Low values are the reliable signal.** A channel that correlates with
    nothing is isolated, floating, or swamped by its own noise, and a threshold
    around 0.6 identifies that dependably.

    **High values are not diagnostic on their own.** It is tempting to read
    near-unity correlation as a salt bridge (conductive gel spreading between
    two electrodes), but on a common-reference recording every channel shares
    the reference signal and any low-frequency drift, which pushes *all*
    pairwise correlations toward 1 — on a real study the median of this metric
    can sit above 0.98, leaving genuine bridges indistinguishable from ordinary
    neighbours. Re-referencing and artifact rejection do not fix it either;
    measuring "electrical distance" (``var(a - b)``, which a bridge collapses)
    fails for the same reason when shared drift dominates.

    So this library does not ship an automatic bridge filter. Use
    :func:`correlation_pairs` to list candidates and confirm them against a
    bipolar derivation, where a bridged pair shows a collapsed amplitude that a
    healthy pair does not.
    """

    name: str = "max_corr"
    l_freq: float = 1.0
    h_freq: float = 45.0

    #: A channel whose amplitude is this far below the montage's typical level
    #: carries no signal. Correlation normalises by standard deviation, so on a
    #: dead channel it divides by numerical residue and returns a *finite*
    #: arbitrary value — which then propagates into other channels' maxima and
    #: makes them depend on floating-point noise. Detected relatively, so the
    #: rule holds whatever units the signal is in.
    dead_rel_tol: float = 1e-9

    def compute_interval(self, ctx, i_start, i_stop):
        seg = ctx.segment(i_start, i_stop, band=(self.l_freq, self.h_freq))
        n_ch = seg.shape[0]
        if n_ch < 2 or seg.shape[1] < 2:
            return np.full(n_ch, np.nan)

        sd = seg.std(axis=1)
        scale = np.median(sd[np.isfinite(sd)]) if np.isfinite(sd).any() else 0.0
        dead = ~np.isfinite(sd) | (sd <= self.dead_rel_tol * scale)

        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(seg)
        C = np.nan_to_num(C, nan=0.0)
        # A dead channel correlates with nothing, and nothing correlates with
        # it — stated explicitly rather than left to whatever the division
        # produced.
        C[dead, :] = 0.0
        C[:, dead] = 0.0
        np.fill_diagonal(C, 0.0)
        return np.abs(C).max(axis=1)


def correlation_pairs(rec, band=(1.0, 45.0), threshold=0.95, top=None):
    """Channel pairs correlated above ``threshold`` — bridge *candidates*.

    Returns ``(channel_a, channel_b, corr, elec_distance)``, where the last
    column is ``var(a - b)`` normalised by the median over all pairs. A true
    bridge drives both toward their extremes, but so does ordinary shared
    reference and drift (see :class:`MaxCorrelation`) — this is a shortlist for
    a human to adjudicate, not a verdict.
    """
    import pandas as pd

    from ..core.context import MetricContext

    ctx = MetricContext(rec, None, ch_type="eeg")
    seg = ctx.segment(0, rec.n_times, band=band)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.corrcoef(seg), nan=0.0)

    n = seg.shape[0]
    ED = np.array([np.var(seg[i][None, :] - seg, axis=1) for i in range(n)])
    iu = np.triu_indices(n, 1)
    med = np.median(ED[iu]) or 1.0

    names = ctx.ch_names
    rows = [(names[i], names[j], float(C[i, j]), float(ED[i, j] / med))
            for i, j in zip(*iu, strict=True) if abs(C[i, j]) >= threshold]
    out = (pd.DataFrame(rows, columns=["channel_a", "channel_b", "corr",
                                       "elec_distance"])
           .sort_values("corr", ascending=False, ignore_index=True))
    return out.head(top) if top else out


max_correlation = MaxCorrelation
