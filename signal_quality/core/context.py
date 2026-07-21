"""Shared, memoized intermediate computations.

The original notebook computed the band-limited signal and two Welch spectra
*once* and shared them across four metrics. Independent metric objects would
naively redo that work per metric — on a whole-night recording that is minutes of
wasted filtering per call. ``MetricContext`` restores the sharing without
reintroducing the coupling: metrics ask for what they need, the context computes
each distinct request once.

Filtering uses ``mne.filter.filter_data``, which takes and returns a plain
ndarray. That is a function call, not a data-model dependency — and it keeps
results numerically identical to the reference analysis this library ports.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


class MetricContext:
    """Per-(recording, grid) cache of intermediate arrays.

    All accessors are keyed by their parameters, so two metrics asking for the
    same band share one computation while a third asking for a different band
    gets its own.
    """

    def __init__(self, rec, grid, ch_type: str | None = None):
        self.rec = rec if ch_type is None else rec.pick(ch_type=ch_type)
        self.grid = grid
        self.sfreq = self.rec.sfreq
        self.ch_names = self.rec.ch_names
        self._cache: dict = {}

    # -- whole-recording arrays ----------------------------------------------
    @property
    def signal(self) -> np.ndarray:
        """Wideband signal in microvolts, ``[n_chan, n_times]``.

        Microvolts rather than volts because every threshold in this field is
        quoted in µV.
        """
        return self._memo("signal_uv", lambda: self.rec.ds["signal"].values * 1e6)

    @property
    def counts(self) -> np.ndarray | None:
        """Raw ADC integers, or None when the source did not preserve them."""
        if not self.rec.has_counts:
            return None
        return self._memo("counts", lambda: self.rec.ds["counts"].values)

    @property
    def covered(self) -> np.ndarray:
        return self._memo("covered", lambda: self.rec.covered)

    def filtered(self, l_freq: float = 1.0, h_freq: float = 45.0) -> np.ndarray:
        """Band-limited signal in µV. Cached per band."""
        def go():
            import mne
            return mne.filter.filter_data(
                self.signal, self.sfreq, l_freq, h_freq, verbose="error")

        return self._memo(("filtered", l_freq, h_freq), go)

    # -- per-interval views ---------------------------------------------------
    def segment(self, i_start, i_stop, band=None, covered_only=True) -> np.ndarray:
        """Signal for one interval, ``[n_chan, n_samples]``.

        ``band`` is ``(l_freq, h_freq)`` or None for wideband. The band-pass is
        applied to the *whole* recording before slicing, never per-window, so
        edge effects do not vary with the grid.
        """
        X = self.signal if band is None else self.filtered(*band)
        seg = X[:, i_start:i_stop]
        if covered_only:
            m = self.covered[i_start:i_stop]
            seg = seg[:, m] if m.size else seg
        return seg

    def welch(self, i_start, i_stop, band=None, nperseg_s: float = 4.0):
        """Welch PSD of one interval -> ``(freqs, power[n_chan, n_freqs])``.

        Cached per (interval, band, nperseg), which is what lets the line-noise
        and EMG metrics share a single spectrum.
        """
        key = ("welch", i_start, i_stop, band, nperseg_s)

        def go():
            seg = self.segment(i_start, i_stop, band=band)
            nperseg = min(int(nperseg_s * self.sfreq), seg.shape[1])
            if nperseg < 8:
                n_f = 1
                return np.zeros(n_f), np.full((seg.shape[0], n_f), np.nan)
            return welch(seg, self.sfreq, nperseg=nperseg, axis=1)

        return self._memo(key, go)

    # -- helpers --------------------------------------------------------------
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @staticmethod
    def band_sum(P, f, lo, hi) -> np.ndarray:
        """Summed power in ``[lo, hi)``. The original's ``bd``/``band`` lambda."""
        return P[:, (f >= lo) & (f < hi)].sum(1)
