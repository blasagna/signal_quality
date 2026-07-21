"""Synthetic recordings with deliberately injected faults.

Real recordings are patient data and stay outside this repo, so the test suite
builds its own signals with known defects. Each fixture injects exactly one kind
of fault, which is what lets a test assert that a metric responds to *that*
fault rather than to something incidental.
"""
from __future__ import annotations

import numpy as np
import pytest

from signal_quality.core.recording import Recording, build_dataset

SFREQ = 250.0
DURATION = 40.0
N = int(SFREQ * DURATION)


def _pink(rng, n, n_ch, exponent: float = 1.0):
    """Noise with a 1/f^exponent *amplitude* spectrum.

    The default gives roughly the spectral slope of real background EEG. This
    matters: with a shallower slope, high-frequency power is a large fraction of
    the total and every channel looks like it is full of muscle artifact, which
    would make the EMG threshold meaningless.
    """
    X = rng.standard_normal((n_ch, n))
    spec = np.fft.rfft(X, axis=1)
    f = np.fft.rfftfreq(n, 1 / SFREQ)
    scale = 1 / np.maximum(f, 0.5) ** exponent
    return np.fft.irfft(spec * scale, n=n, axis=1)


@pytest.fixture
def clean_rec():
    """Six correlated 1/f channels, no injected fault."""
    rng = np.random.default_rng(0)
    common = _pink(rng, N, 1)
    X = _pink(rng, N, 6) * 0.5 + common * 1.2
    X *= 20e-6 / X.std()
    names = ["C3", "C4", "O1", "O2", "F3", "F4"]
    ds = build_dataset(X, SFREQ, names, ["eeg"] * 6,
                       covered=np.ones(N, dtype=bool))
    return Recording(ds, None, provenance={})


def _rec_from(X, names, counts=None, covered=None, factor_uV=None):
    ds = build_dataset(X, SFREQ, names, ["eeg"] * len(names), counts=counts,
                       covered=(np.ones(X.shape[1], dtype=bool)
                                if covered is None else covered),
                       factor_uV=factor_uV)
    return Recording(ds, None, provenance={})


@pytest.fixture
def faulty_rec():
    """One channel per fault, plus two clean references.

    ``FLAT`` is dead, ``LINE`` carries a large mains tone, ``BRIDGE`` is a near
    copy of ``C3``, ``LOUD`` is an order of magnitude too big.
    """
    rng = np.random.default_rng(1)
    common = _pink(rng, N, 1)
    base = _pink(rng, N, 6) * 0.5 + common * 1.2
    base *= 20e-6 / base.std()
    names = ["C3", "C4", "FLAT", "LINE", "BRIDGE", "LOUD"]

    X = base.copy()
    X[2] = 0.0                                              # flat / dead
    t = np.arange(N) / SFREQ
    X[3] = X[3] + 400e-6 * np.sin(2 * np.pi * 60 * t)       # mains pickup
    X[4] = X[0] * 0.99 + 0.02 * X[4]                        # salt bridge to C3
    X[5] = X[5] * 25                                        # amplitude outlier
    return _rec_from(X, names)


@pytest.fixture
def clipping_rec():
    """A recording with raw ADC counts, one channel pinned at the rail."""
    rng = np.random.default_rng(2)
    base = _pink(rng, N, 3)
    base *= 20e-6 / base.std()
    factor_uV = np.full(3, 0.2658386864277569)
    counts = (base * 1e6 / factor_uV[:, None]).astype(np.int32)
    counts[1, ::7] = 131071                                  # rail, ~14% of samples
    signal = counts.astype(np.float64) * factor_uV[:, None] * 1e-6
    return _rec_from(signal, ["C3", "RAILED", "C4"], counts=counts,
                     factor_uV=factor_uV)


@pytest.fixture
def gapped_rec():
    """A clean recording with a 10 s hole in the middle."""
    rng = np.random.default_rng(3)
    X = _pink(rng, N, 3)
    X *= 20e-6 / X.std()
    covered = np.ones(N, dtype=bool)
    i0, i1 = int(15 * SFREQ), int(25 * SFREQ)
    covered[i0:i1] = False
    X[:, i0:i1] = 0.0
    return _rec_from(X, ["C3", "C4", "O1"], covered=covered)


@pytest.fixture
def stamp_tables():
    """Hand-built .etc/.stc tables: monotonic, and a corrupted variant."""
    dt_etc = np.dtype([("offset", "<i4"), ("samplestamp", "<i4"),
                       ("sample_num", "<i4"), ("sample_span", "<i2"),
                       ("unknown", "<i2")])
    span = 100
    n_pkt = 20
    ok = np.zeros(n_pkt, dtype=dt_etc)
    ok["samplestamp"] = np.arange(n_pkt) * span
    ok["sample_span"] = span

    backwards = ok.copy()
    backwards["samplestamp"][10] = 300          # jumps back in time

    jumpy = ok.copy()
    jumpy["samplestamp"][12:] += 500            # irregular period at one boundary

    overlapping = ok.copy()
    overlapping["samplestamp"][8:] -= 40        # packet starts before prev ends
    return dict(ok=ok, backwards=backwards, jumpy=jumpy, overlapping=overlapping)
