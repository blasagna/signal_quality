"""Amplitude-domain quality metrics.

Formulas are preserved from the reference analysis so results stay comparable;
every constant that was inline there is a parameter here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.metric import Metric


@dataclass
class RMS(Metric):
    """Broadband amplitude: standard deviation of the band-limited signal, µV.

    Low outliers are attenuated or poorly-contacting electrodes; high outliers
    are noisy ones. Compare across channels with a ``RobustZ`` filter rather
    than an absolute threshold — absolute µV varies with montage and subject.
    """

    name: str = "rms"
    l_freq: float = 1.0
    h_freq: float = 45.0

    def compute_interval(self, ctx, i_start, i_stop):
        seg = ctx.segment(i_start, i_stop, band=(self.l_freq, self.h_freq))
        if seg.shape[1] < 2:
            return np.full(seg.shape[0], np.nan)
        return seg.std(axis=1)


@dataclass
class FlatFraction(Metric):
    """Fraction of short windows whose amplitude is essentially zero.

    A dead channel, a disconnected lead, or a signal that never varies. The
    default 0.5 s window / 0.5 µV threshold comes from the reference analysis.
    """

    name: str = "flat_frac"
    win: float = 0.5
    thresh: float = 0.5
    l_freq: float = 1.0
    h_freq: float = 45.0

    def compute_interval(self, ctx, i_start, i_stop):
        seg = ctx.segment(i_start, i_stop, band=(self.l_freq, self.h_freq))
        wl = max(1, int(self.win * ctx.sfreq))
        n = seg.shape[1] // wl
        if n == 0:
            return np.full(seg.shape[0], np.nan)
        blocks = seg[:, :n * wl].reshape(seg.shape[0], n, wl)
        return (blocks.std(axis=2) < self.thresh).mean(axis=1)


@dataclass
class ClipFraction(Metric):
    """Percentage of samples sitting at the converter rail.

    Needs raw ADC integers: once converted to volts the rail is no longer
    identifiable as a distinguished value, so this returns NaN for sources that
    only store scaled data.
    """

    name: str = "clip_pct"
    requires: tuple = ("counts",)
    rail: int = 131071  # 2**17 - 1

    def compute_interval(self, ctx, i_start, i_stop):
        counts = ctx.counts[:, i_start:i_stop]
        m = ctx.covered[i_start:i_stop]
        if m.any():
            counts = counts[:, m]
        if counts.shape[1] == 0:
            return np.full(counts.shape[0], np.nan)
        return 100.0 * (np.abs(counts) >= self.rail).mean(axis=1)


@dataclass
class PeakToPeak(Metric):
    """Peak-to-peak amplitude, µV.

    Over a fine grid this is the movement/artifact trace: the fraction of short
    windows under a threshold is the "clean epoch" measure of the reference
    analysis.
    """

    name: str = "p2p"
    l_freq: float = 1.0
    h_freq: float = 45.0

    def compute_interval(self, ctx, i_start, i_stop):
        seg = ctx.segment(i_start, i_stop, band=(self.l_freq, self.h_freq))
        if seg.shape[1] == 0:
            return np.full(seg.shape[0], np.nan)
        return np.ptp(seg, axis=1)


rms = RMS
flat_fraction = FlatFraction
clip_fraction = ClipFraction
p2p = PeakToPeak
