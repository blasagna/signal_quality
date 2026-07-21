"""Frequency-domain quality metrics.

All of these carry a ``min_analysis_s``. Frequency resolution is set by the
length of the analysed window, and a short window cannot separate a narrow peak
from its own spectral leakage. Measured on a real study: a mains-noise ratio of
963 over the whole recording reads **5.1** if computed from a bare 1-second
window, because leakage from the 60 Hz peak floods the 61–65 Hz reference band —
the metric silently stops measuring what it claims to. At 2 s it reads ~13,000
and by 4 s it has plateaued.

So the interval a value is *attributed to* is decoupled from the window it is
*computed from*: the PSD comes from a wider centered window, while the flag
still lands on the fine interval. The cost is temporal smearing — a one-second
burst of line noise is spread over ``min_analysis_s`` — so **onset timing from
spectral flags is approximate**. Amplitude-domain metrics keep exact bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.metric import Metric

#: Enough to resolve a mains peak from its neighbours; see module docstring.
DEFAULT_MIN_ANALYSIS_S = 4.0

#: Power this far below the montage's typical level means the channel carries
#: no signal at all.
DEAD_REL_TOL = 1e-12


def _dead(P, f, lo=0.0, hi=np.inf, rel_tol=DEAD_REL_TOL):
    """Channels with negligible power, relative to the rest of the montage.

    Every metric here is a *ratio* of band powers. On a dead channel both parts
    are numerical residue, so the ratio is arbitrary — it varies with unrelated
    implementation details and can land anywhere in range. Reporting NaN says
    the honest thing: there is no signal to characterise.
    """
    total = P[:, (f >= lo) & (f < hi)].sum(1)
    finite = np.isfinite(total)
    scale = np.median(total[finite]) if finite.any() else 0.0
    return ~finite | (total <= rel_tol * scale)


@dataclass
class LineRatio(Metric):
    """Mains-interference ratio: power at the line frequency over its neighbours.

    A poorly-contacting, high-impedance or floating electrode acts as an antenna
    for mains noise, which makes this the single strongest cue for bad contact.
    Taking a *ratio* against adjacent bands rather than absolute line power makes
    it insensitive to overall channel gain.

    ``f0`` defaults to None, meaning "use the recording's ``line_freq``" — 60 Hz
    in the Americas, 50 Hz across most of the rest of the world. Computed on
    wideband data, since a 1–45 Hz band-pass would remove the very thing being
    measured.
    """

    name: str = "line_ratio"
    f0: float | None = None
    half_width: float = 1.0
    side_width: float = 4.0
    min_analysis_s: float = DEFAULT_MIN_ANALYSIS_S

    def compute_interval(self, ctx, i_start, i_stop):
        f0 = self.f0 if self.f0 is not None else ctx.rec.line_freq
        f, P = ctx.welch(i_start, i_stop, band=None,
                         min_analysis_s=self.min_analysis_s)
        if P.shape[1] < 2:
            return np.full(P.shape[0], np.nan)
        hw, sw = self.half_width, self.side_width
        at = ctx.band_sum(P, f, f0 - hw, f0 + hw)
        below = ctx.band_sum(P, f, f0 - hw - sw, f0 - hw)
        above = ctx.band_sum(P, f, f0 + hw, f0 + hw + sw)
        out = at / (below + above + 1e-30)
        return np.where(_dead(P, f, f0 - hw - sw, f0 + hw + sw), np.nan, out)


@dataclass
class EMGFraction(Metric):
    """Percentage of in-band power above ``fmin`` — muscle contamination.

    High-frequency power in a scalp recording is mostly EMG from a tense or
    moving subject rather than cerebral activity.
    """

    name: str = "emg_pct"
    fmin: float = 25.0
    fmax: float = 45.0
    band_lo: float = 1.0
    min_analysis_s: float = DEFAULT_MIN_ANALYSIS_S

    def analysis_band(self):
        return (self.band_lo, self.fmax)

    def compute_interval(self, ctx, i_start, i_stop):
        f, P = ctx.welch(i_start, i_stop, band=self.analysis_band(),
                         min_analysis_s=self.min_analysis_s)
        if P.shape[1] < 2:
            return np.full(P.shape[0], np.nan)
        total = ctx.band_sum(P, f, self.band_lo, self.fmax)
        out = 100.0 * ctx.band_sum(P, f, self.fmin, self.fmax) / (total + 1e-30)
        return np.where(_dead(P, f, self.band_lo, self.fmax), np.nan, out)


@dataclass
class BandPower(Metric):
    """Summed power in an arbitrary band.

    Generic building block — instantiate it more than once (with distinct
    ``name``s) to get, say, theta and alpha columns you can then ratio in the
    joined table.
    """

    name: str = "band_power"
    fmin: float = 8.0
    fmax: float = 13.0
    band: tuple | None = (1.0, 45.0)
    min_analysis_s: float = DEFAULT_MIN_ANALYSIS_S

    def analysis_band(self):
        return self.band

    def compute_interval(self, ctx, i_start, i_stop):
        f, P = ctx.welch(i_start, i_stop, band=self.band,
                         min_analysis_s=self.min_analysis_s)
        if P.shape[1] < 2:
            return np.full(P.shape[0], np.nan)
        return ctx.band_sum(P, f, self.fmin, self.fmax)   # absolute: 0 is valid


line_ratio = LineRatio
emg_fraction = EMGFraction
band_power = BandPower
