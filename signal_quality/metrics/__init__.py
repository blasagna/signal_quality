"""Individual quality metrics.

Nothing here imports MNE's data model: metrics see xarray and numpy only, which
is what keeps the library usable on non-EEG signals.
"""
from .amplitude import (ClipFraction, FlatFraction, PeakToPeak, RMS,
                        clip_fraction, flat_fraction, p2p, rms)
from .spatial import MaxCorrelation, correlation_pairs, max_correlation
from .spectral import (BandPower, EMGFraction, LineRatio, band_power,
                       emg_fraction, line_ratio)
from . import integrity

#: The per-channel metric set ported from the reference EEG analysis.
DEFAULT_METRICS = [RMS, LineRatio, EMGFraction, MaxCorrelation, FlatFraction,
                   ClipFraction]

__all__ = [
    "RMS", "FlatFraction", "ClipFraction", "PeakToPeak",
    "LineRatio", "EMGFraction", "BandPower",
    "MaxCorrelation", "correlation_pairs",
    "rms", "flat_fraction", "clip_fraction", "p2p",
    "line_ratio", "emg_fraction", "band_power", "max_correlation",
    "integrity", "DEFAULT_METRICS",
]
