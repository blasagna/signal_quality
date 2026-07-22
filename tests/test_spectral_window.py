"""Spectral metrics must not degrade silently at short intervals.

Frequency resolution is set by the analysed window. On a 1-second window the
mains peak cannot be separated from its own leakage, and the line-noise ratio
for a badly-contacting electrode measured **5.1** instead of the ~963 seen over
the whole recording — no error, no warning, just a metric that had stopped
measuring what it claims to. These tests pin the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

import signal_quality as sq
from signal_quality import metrics as M
from signal_quality.core.recording import Recording, build_dataset

SF = 250.0
DUR = 60.0


@pytest.fixture(scope="module")
def mains_rec():
    """Four clean channels; one drenched in mains noise."""
    n = int(SF * DUR)
    rng = np.random.default_rng(0)
    t = np.arange(n) / SF
    spec = np.fft.rfft(rng.standard_normal((4, n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / SF)
    X = np.fft.irfft(spec / np.maximum(f, 0.5), n=n, axis=1)
    X *= 20e-6 / X.std(axis=1, keepdims=True)
    X[1] += 300e-6 * np.sin(2 * np.pi * 60 * t)
    return Recording(build_dataset(X, SF, ["A", "NOISY", "B", "C"], ["eeg"] * 4))


def test_one_second_intervals_still_resolve_mains(mains_rec):
    mf = sq.compute([M.LineRatio()], mains_rec)  # default 1 s grid
    per_ch = mf.table["line_ratio"].groupby(level="channel").median()

    assert per_ch["NOISY"] > 1000, (
        f"line_ratio collapsed to {per_ch['NOISY']:.1f} — the 1-second window is "
        f"being analysed at its own resolution instead of a widened one"
    )
    assert per_ch.drop("NOISY").max() < 100


def test_without_widening_the_metric_collapses(mains_rec):
    """Demonstrates the failure the widening exists to prevent."""
    bare = sq.compute([M.LineRatio(min_analysis_s=0.0)], mains_rec)
    widened = sq.compute([M.LineRatio(min_analysis_s=4.0)], mains_rec)

    b = bare.table["line_ratio"].groupby(level="channel").median()["NOISY"]
    w = widened.table["line_ratio"].groupby(level="channel").median()["NOISY"]
    assert b < 100 < w
    assert w > 50 * b


def test_flag_lands_on_the_right_channel_at_one_second(mains_rec):
    mf = sq.compute([M.LineRatio(), M.RMS(), M.PeakToPeak()], mains_rec)
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    line = flags[flags["flag"] == "LINE_NOISE"]
    assert len(line), "LINE_NOISE never fired at the default 1-second grid"
    assert set(line["channel"]) == {"NOISY"}


def test_analysis_window_is_centered_and_clamped(mains_rec):
    from signal_quality.core.context import MetricContext

    ctx = MetricContext(mains_rec, None)
    sf, n = ctx.sfreq, ctx.n_times

    a, b = ctx.analysis_bounds(int(30 * sf), int(31 * sf), 4.0)
    assert b - a == int(4 * sf)
    assert a < int(30 * sf) and b > int(31 * sf)  # centered

    a, b = ctx.analysis_bounds(0, int(1 * sf), 4.0)  # at the start
    assert (a, b - a) == (0, int(4 * sf))
    a, b = ctx.analysis_bounds(n - int(1 * sf), n, 4.0)  # at the end
    assert b == n and b - a == int(4 * sf)


def test_spectra_are_nan_when_the_window_is_mostly_gap(gapped_rec):
    """Better an admitted NaN than a number computed at a resolution the caller
    never asked for."""
    mf = sq.compute([M.LineRatio()], gapped_rec)
    inside = mf.table["coverage"] <= 0
    assert mf.table.loc[inside[inside].index, "line_ratio"].isna().all()
    outside = mf.table["coverage"] >= 1.0
    assert mf.table.loc[outside[outside].index, "line_ratio"].notna().any()


def test_amplitude_metrics_keep_exact_interval_bounds(mains_rec):
    """Only spectral metrics widen; an amplitude metric must see its interval
    and nothing else, or artifact onsets would smear too."""
    assert M.RMS().required_pad_s(SF) > 0  # filter context only
    assert getattr(M.RMS(), "min_analysis_s", 0.0) == 0.0
    assert getattr(M.PeakToPeak(), "min_analysis_s", 0.0) == 0.0
    assert M.LineRatio().min_analysis_s == 4.0
