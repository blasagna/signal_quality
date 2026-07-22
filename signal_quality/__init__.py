"""Modular signal-quality metrics.

Each quality check is an independent metric computed over a shared grid of
intervals; metrics join into one table; filters are defined separately and
applied to that table. Typical use::

    import signal_quality as sq

    rec  = sq.load("/path/to/study")
    grid = sq.IntervalGrid.whole(rec)
    mf   = sq.compute([sq.metrics.RMS(), sq.metrics.LineRatio(),
                       sq.metrics.MaxCorrelation()], rec, grid)

    issues = sq.check_integrity(rec)                     # gaps, clock, alignment
    flags  = sq.apply_filters(mf, sq.DEFAULT_FILTERS)    # policy, applied after
    print(sq.verdict(flags, mf))                         # pass mf so clean cells show too
"""

from . import metrics, montage, viz
from .core.frame import MetricFrame
from .core.intervals import IntervalGrid
from .core.metric import Metric, compute
from .export import to_annotations
from .filters import (
    DEFAULT_FILTERS,
    WHOLE_RECORDING_FILTERS,
    Filter,
    RobustZ,
    Threshold,
    apply_filters,
    bad_segments,
    channel_summary,
    verdict,
)
from .io.load import load
from .metrics.integrity import (
    channel_alignment,
    check_integrity,
    coverage_gaps,
    timestamp_anomalies,
)
from .report import QualityReport, assess
from .synthetic import make_demo_recording

__all__ = [
    "load",
    "assess",
    "QualityReport",
    "IntervalGrid",
    "Metric",
    "MetricFrame",
    "compute",
    "Filter",
    "Threshold",
    "RobustZ",
    "apply_filters",
    "verdict",
    "channel_summary",
    "bad_segments",
    "to_annotations",
    "WHOLE_RECORDING_FILTERS",
    "DEFAULT_FILTERS",
    "check_integrity",
    "coverage_gaps",
    "timestamp_anomalies",
    "channel_alignment",
    "make_demo_recording",
    "metrics",
    "montage",
    "viz",
]
__version__ = "0.1.0"
