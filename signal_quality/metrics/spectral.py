"""Frequency-domain quality metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.metric import Metric


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

    def compute_interval(self, ctx, i_start, i_stop):
        f0 = self.f0 if self.f0 is not None else ctx.rec.line_freq
        f, P = ctx.welch(i_start, i_stop, band=None)
        if P.shape[1] < 2:
            return np.full(P.shape[0], np.nan)
        hw, sw = self.half_width, self.side_width
        at = ctx.band_sum(P, f, f0 - hw, f0 + hw)
        below = ctx.band_sum(P, f, f0 - hw - sw, f0 - hw)
        above = ctx.band_sum(P, f, f0 + hw, f0 + hw + sw)
        return at / (below + above + 1e-30)


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

    def compute_interval(self, ctx, i_start, i_stop):
        f, P = ctx.welch(i_start, i_stop, band=(self.band_lo, self.fmax))
        if P.shape[1] < 2:
            return np.full(P.shape[0], np.nan)
        total = ctx.band_sum(P, f, self.band_lo, self.fmax)
        return 100.0 * ctx.band_sum(P, f, self.fmin, self.fmax) / (total + 1e-30)


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

    def compute_interval(self, ctx, i_start, i_stop):
        f, P = ctx.welch(i_start, i_stop, band=self.band)
        if P.shape[1] < 2:
            return np.full(P.shape[0], np.nan)
        return ctx.band_sum(P, f, self.fmin, self.fmax)


line_ratio = LineRatio
emg_fraction = EMGFraction
band_power = BandPower
