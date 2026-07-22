"""Shared, memoized intermediate computations.

Metrics ask for what they need — band-limited signal, a spectrum — and the
context computes each distinct request once. Two things it must get right:

**Scope.** The cached arrays cover a *view*: either the whole recording (for
standalone use and plotting) or one padded block (during ``compute``). Metrics
always address samples in absolute recording coordinates; the context maps them
into the view. This is what lets block processing bound memory without any
metric knowing it is happening.

**Analysis windows.** The interval a value is *attributed to* need not be the
window it is *computed from*. A spectral metric on a 1-second interval has only
1 Hz of frequency resolution, which is not enough to separate a mains peak from
its own leakage — measured on a real study, a line-noise ratio of 963 collapses
to 5. Such metrics request a wider centered analysis window and keep the fine
time attribution.

Filtering uses ``mne.filter.filter_data``, which takes and returns a plain
ndarray. That is a function call, not a data-model dependency — and it keeps
results numerically identical to the reference analysis this library ports.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


class MetricContext:
    """Per-(recording, view) cache of intermediate arrays."""

    def __init__(self, rec, grid=None, ch_type: str | None = None):
        self.rec = rec if ch_type is None else rec.pick(ch_type=ch_type)
        self.grid = grid
        self.sfreq = self.rec.sfreq
        self.ch_names = self.rec.ch_names
        self.n_times = self.rec.n_times
        self._view = (0, self.n_times)
        self._cache: dict = {}

    # -- view management ------------------------------------------------------
    @property
    def view(self) -> tuple[int, int]:
        """Sample span currently materialised, in recording coordinates."""
        return self._view

    def set_view(self, start: int, stop: int) -> None:
        """Restrict cached arrays to ``[start, stop)`` and drop everything else.

        Called once per block. Dropping the cache here is the point: keeping
        per-interval spectra alive across a whole recording costs far more than
        recomputing them, because nothing ever revisits an interval.
        """
        start, stop = int(max(0, start)), int(min(self.n_times, stop))
        if (start, stop) != self._view:
            self._cache.clear()
            self._view = (start, stop)

    def release(self) -> None:
        """Drop cached arrays, keeping the view."""
        self._cache.clear()

    def _slice(self, i_start: int, i_stop: int) -> slice:
        v0, v1 = self._view
        if i_start < v0 or i_stop > v1:
            raise ValueError(
                f"samples [{i_start}, {i_stop}) fall outside the loaded view "
                f"[{v0}, {v1}) — the block was not padded enough for this metric"
            )
        return slice(i_start - v0, i_stop - v0)

    # -- view-scoped arrays ---------------------------------------------------
    @property
    def signal(self) -> np.ndarray:
        """Wideband signal in microvolts over the current view.

        Microvolts rather than volts because every threshold in this field is
        quoted in µV.
        """

        def go():
            v0, v1 = self._view
            return self.rec.ds["signal"].values[:, v0:v1] * 1e6

        return self._memo("signal_uv", go)

    @property
    def counts(self) -> np.ndarray | None:
        """Raw ADC integers over the view, or None if the source lacks them."""
        if not self.rec.has_counts:
            return None

        def go():
            v0, v1 = self._view
            return self.rec.ds["counts"].values[:, v0:v1]

        return self._memo("counts", go)

    @property
    def covered(self) -> np.ndarray:
        """Coverage mask for the whole recording (cheap; kept unsliced)."""
        return self._memo("covered", lambda: self.rec.covered)

    def filtered(self, l_freq: float = 1.0, h_freq: float = 45.0) -> np.ndarray:
        """Band-limited signal over the view, in µV. Cached per band."""

        def go():
            import mne

            return mne.filter.filter_data(self.signal, self.sfreq, l_freq, h_freq, verbose="error")

        return self._memo(("filtered", l_freq, h_freq), go)

    # -- interval views -------------------------------------------------------
    def analysis_bounds(
        self, i_start: int, i_stop: int, min_analysis_s: float = 0.0
    ) -> tuple[int, int]:
        """Widen an interval to at least ``min_analysis_s``, centered.

        Clamped to the recording, so intervals at the very start or end are
        shifted inward rather than truncated — a truncated window would quietly
        analyse at worse resolution than asked for.
        """
        need = int(round(min_analysis_s * self.sfreq))
        have = i_stop - i_start
        if need <= have:
            return i_start, i_stop
        extra = need - have
        a = i_start - extra // 2
        b = a + need
        if a < 0:
            a, b = 0, min(self.n_times, need)
        if b > self.n_times:
            b = self.n_times
            a = max(0, b - need)
        return a, b

    def segment(self, i_start, i_stop, band=None, covered_only=True) -> np.ndarray:
        """Signal for one span, ``[n_chan, n_samples]``.

        ``band`` is ``(l_freq, h_freq)`` or None for wideband. The band-pass is
        applied to the whole padded view before slicing, never to the bare
        interval, so a window's filtering does not depend on the grid.
        """
        X = self.signal if band is None else self.filtered(*band)
        seg = X[:, self._slice(i_start, i_stop)]
        if covered_only:
            m = self.covered[i_start:i_stop]
            seg = seg[:, m] if m.size else seg
        return seg

    def counts_segment(self, i_start, i_stop, covered_only=True):
        """Raw ADC integers for one span, or None when unavailable.

        Goes through the same view mapping as :meth:`segment` — metrics address
        samples in recording coordinates and must never index the cached arrays
        directly, which are only block-sized.
        """
        if self.counts is None:
            return None
        seg = self.counts[:, self._slice(i_start, i_stop)]
        if covered_only:
            m = self.covered[i_start:i_stop]
            seg = seg[:, m] if m.size else seg
        return seg

    def coverage(self, i_start: int, i_stop: int) -> float:
        m = self.covered[i_start:i_stop]
        return float(m.mean()) if m.size else 0.0

    def welch(
        self, i_start, i_stop, band=None, nperseg_s: float = 4.0, min_analysis_s: float = 0.0
    ):
        """Welch PSD for an interval -> ``(freqs, power[n_chan, n_freqs])``.

        The interval is widened to ``min_analysis_s`` first, then gap samples
        are dropped. If too few covered samples remain to deliver the requested
        resolution, returns NaN rather than a spectrum quietly computed at a
        coarser resolution than asked for — the failure this guards against is
        exactly the one that made a line-noise ratio of 963 read as 5.

        The test is on *covered samples*, not on the covered fraction: a
        whole-recording window with a 13% gap still has ample data, while a 4 s
        window sitting inside a gap does not.
        """
        a, b = self.analysis_bounds(i_start, i_stop, min_analysis_s)
        key = ("welch", a, b, band, nperseg_s)

        def go():
            n_ch = len(self.ch_names)
            nan = (np.zeros(1), np.full((n_ch, 1), np.nan))
            seg = self.segment(a, b, band=band)
            need = max(8, int(round(min_analysis_s * self.sfreq)))
            if seg.shape[1] < need:
                return nan
            nperseg = min(int(nperseg_s * self.sfreq), seg.shape[1])
            if nperseg < 8:
                return nan
            return welch(seg, self.sfreq, nperseg=nperseg, axis=1)

        return self._memo(key, go)

    # -- helpers --------------------------------------------------------------
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @staticmethod
    def band_sum(P, f, lo, hi) -> np.ndarray:
        """Summed power in ``[lo, hi)``."""
        return P[:, (f >= lo) & (f < hi)].sum(1)
