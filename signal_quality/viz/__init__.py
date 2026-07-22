"""Plots. Every function takes computed results and shows what was flagged."""

from .heatmap import plot_flag_timeline, plot_quality_heatmap
from .psd import plot_good_bad_psd, plot_psd_examples
from .timeline import plot_availability, plot_clean_fraction, plot_metric_trend
from .topomap import (
    plot_contact_quality,
    plot_metric_topomap,
    plot_pct_bad_topomap,
    plot_verdict_topomap,
)
from .traces import plot_channel_snippet, plot_flagged_examples, plot_overview

__all__ = [
    "plot_quality_heatmap",
    "plot_flag_timeline",
    "plot_availability",
    "plot_metric_trend",
    "plot_clean_fraction",
    "plot_metric_topomap",
    "plot_verdict_topomap",
    "plot_pct_bad_topomap",
    "plot_contact_quality",
    "plot_good_bad_psd",
    "plot_psd_examples",
    "plot_overview",
    "plot_channel_snippet",
    "plot_flagged_examples",
]
