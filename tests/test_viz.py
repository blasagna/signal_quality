"""Plot smoke tests.

These do not check that a figure *looks* right — only that each plotting path
runs on realistic inputs and puts something on the axes. Plotting code breaks
silently and is usually only exercised from a notebook, so cheap coverage here
is worth having.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import signal_quality as sq
from signal_quality import metrics as M


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def scalp_rec(faulty_rec):
    """The faulty fixture relabelled to real 10-20 sites, so it can be placed."""
    from signal_quality.core.recording import Recording, build_dataset

    # Positional, so the fixture's faults land on known sites:
    #   C3=clean C4=clean O1=flat O2=mains F3=bridge-to-C3 F4=loud
    names = ["C3", "C4", "O1", "O2", "F3", "F4"]
    ds = build_dataset(faulty_rec.ds["signal"].values, faulty_rec.sfreq, names,
                       ["eeg"] * 6)
    return Recording(ds)


@pytest.fixture
def computed(scalp_rec):
    mf = sq.compute([M.RMS(), M.LineRatio(), M.MaxCorrelation(),
                     M.FlatFraction(), M.EMGFraction()],
                    scalp_rec, sq.IntervalGrid.whole(scalp_rec))
    flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)
    verdicts = sq.verdict(flags, channels=scalp_rec.ch_names)
    return mf, flags, verdicts


def test_availability_marks_the_gap(gapped_rec):
    ax = sq.viz.plot_availability(gapped_rec, sq.check_integrity(gapped_rec))
    assert ax.collections or ax.containers
    assert ax.get_xlim()[1] == pytest.approx(gapped_rec.duration, rel=0.01)


def test_contact_quality_figure(scalp_rec, computed):
    mf, _, verdicts = computed
    fig = sq.viz.plot_contact_quality(mf, verdicts, metric="line_ratio", log=True)
    assert len(fig.axes) >= 2


def test_verdict_topomap_labels_every_placed_channel(computed):
    _, _, verdicts = computed
    ax = sq.viz.plot_verdict_topomap(verdicts)
    labels = {t.get_text() for t in ax.texts}
    assert {"C3", "O1", "F4"} <= labels


def test_good_bad_psd_ranks_worst_first(scalp_rec, computed):
    mf, flags, _ = computed
    ax = sq.viz.plot_good_bad_psd(scalp_rec, flags, flag="LINE_NOISE")
    labelled = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("O2" in t for t in labelled)      # O2 carries the injected tone
    assert ax.lines


def test_good_bad_psd_refuses_an_absent_flag(scalp_rec, computed):
    _, flags, _ = computed
    with pytest.raises(ValueError, match="no channels flagged"):
        sq.viz.plot_good_bad_psd(scalp_rec, flags, flag="NOT_A_FLAG")


def test_metric_trend_and_clean_fraction(scalp_rec):
    grid = sq.IntervalGrid.fixed(scalp_rec, 5.0)
    wmf = sq.compute([M.RMS(), M.PeakToPeak(), M.LineRatio()], scalp_rec, grid)
    ax = sq.viz.plot_metric_trend(wmf, "line_ratio", rec=scalp_rec, threshold=300)
    assert len(ax.lines[0].get_xdata()) == len(grid)
    ax2 = sq.viz.plot_clean_fraction(wmf, rec=scalp_rec, threshold=150.0)
    assert ax2.get_ylim() == (0, 101)


def test_snippet_plot_and_unknown_channel(scalp_rec):
    ax = sq.viz.plot_channel_snippet(scalp_rec, ["C3", "C4"], 0.0, 5.0)
    assert len(ax.lines) == 2
    with pytest.raises(KeyError):
        sq.viz.plot_channel_snippet(scalp_rec, ["NOPE"], 0.0, 5.0)


def test_snippet_picks_a_covered_window(gapped_rec):
    """The default window must not land inside the recording gap."""
    from signal_quality.viz.traces import _quiet_window

    t0 = _quiet_window(gapped_rec, 5.0)
    i0 = int(t0 * gapped_rec.sfreq)
    assert gapped_rec.covered[i0:i0 + int(5 * gapped_rec.sfreq)].all()
