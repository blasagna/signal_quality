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
    """Whole-recording view, for the plots that need one row per channel."""
    mf = sq.compute([M.RMS(), M.LineRatio(), M.MaxCorrelation(),
                     M.FlatFraction(), M.EMGFraction()],
                    scalp_rec, sq.IntervalGrid.whole(scalp_rec))
    flags = sq.apply_filters(mf, sq.WHOLE_RECORDING_FILTERS)
    verdicts = sq.verdict(flags, mf)
    return mf, flags, sq.channel_summary(verdicts, bad_time_frac=0.0)


def test_availability_marks_the_gap(gapped_rec):
    ax = sq.viz.plot_availability(gapped_rec, sq.check_integrity(gapped_rec))
    assert ax.collections or ax.containers
    assert ax.get_xlim()[1] == pytest.approx(gapped_rec.duration, rel=0.01)


def test_contact_quality_figure(scalp_rec, computed):
    mf, _, summary = computed
    fig = sq.viz.plot_contact_quality(mf, summary, metric="line_ratio", log=True)
    assert len(fig.axes) >= 2


def test_verdict_topomap_rejects_per_interval_input(scalp_rec):
    """A head plot can show one value per electrode; which value that should be
    is a policy question, so it must be rolled up first rather than guessed."""
    mf = sq.compute([M.RMS(), M.PeakToPeak()], scalp_rec)
    v = sq.verdict(sq.apply_filters(mf, sq.DEFAULT_FILTERS), mf)
    with pytest.raises(TypeError, match="channel_summary"):
        sq.viz.plot_verdict_topomap(v)


def test_pct_bad_topomap(scalp_rec):
    mf = sq.compute([M.RMS(), M.PeakToPeak(), M.FlatFraction()], scalp_rec)
    v = sq.verdict(sq.apply_filters(mf, sq.DEFAULT_FILTERS), mf)
    ax = sq.viz.plot_pct_bad_topomap(sq.channel_summary(v))
    assert "%" in ax.figure.axes[-1].get_ylabel()


def test_log_topomap_survives_a_dead_channel():
    """A dead channel has no spectral ratio at all (NaN), and a near-zero one is
    -inf once logged. Naive colour limits would stretch the scale until every
    other electrode rendered as one flat colour."""
    rec, _ = sq.make_demo_recording()
    mf = sq.compute([M.LineRatio(), M.FlatFraction()], rec,
                    sq.IntervalGrid.whole(rec))
    assert mf.table["line_ratio"].isna().any()          # the dead electrode

    ax = sq.viz.plot_metric_topomap(mf, "line_ratio", log=True)
    lo, hi = ax.images[0].get_clim() if ax.images else ax.collections[0].get_clim()
    assert np.isfinite([lo, hi]).all()
    # Real line ratios span ~1 to ~3000, i.e. 0..3.5 in log10. A scale blown out
    # by the dead channel would be far wider than this.
    assert (hi - lo) < 8


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


# --- y-axis scale annotation ------------------------------------------------

def test_trend_ylabel_states_the_observed_range(scalp_rec):
    grid = sq.IntervalGrid.fixed(scalp_rec, 5.0)
    wmf = sq.compute([M.RMS()], scalp_rec, grid)
    ax = sq.viz.plot_metric_trend(wmf, "rms")
    assert "range:" in ax.get_ylabel()


def test_stacked_plots_carry_a_scale_bar(scalp_rec):
    """Channel-name y ticks say nothing about amplitude; the bar must."""
    ax = sq.viz.plot_channel_snippet(scalp_rec, ["C3", "C4"], 0.0, 5.0, step=150.0)
    assert "150" in ax.get_ylabel()                    # lane spacing stated
    assert any("150" in t.get_text() and "µV" in t.get_text() for t in ax.texts)
    assert "peak" in ax.get_title()


def test_psd_ylabel_states_the_range(scalp_rec, computed):
    _, flags, _ = computed
    ax = sq.viz.plot_good_bad_psd(scalp_rec, flags, flag="LINE_NOISE")
    assert "range:" in ax.get_ylabel()


# --- whole-recording overview -----------------------------------------------

def test_overview_draws_every_channel(scalp_rec):
    ax = sq.viz.plot_overview(scalp_rec)
    assert len(ax.collections) >= len(scalp_rec.ch_names)
    assert [t.get_text() for t in ax.get_yticklabels()] == scalp_rec.ch_names
    assert "µV" in ax.get_ylabel()


def test_overview_envelope_preserves_brief_excursions():
    """A spike far shorter than one pixel column must still be visible.

    Plain decimation would step straight over it, which is precisely the kind of
    event this plot exists to reveal.
    """
    import numpy as np

    from signal_quality.core.recording import Recording, build_dataset

    sf, n = 250.0, 40_000
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, n)) * 20e-6
    X[0, 12_345] = 5_000e-6                       # one enormous sample
    rec = Recording(build_dataset(X, sf, ["A", "B"], ["eeg", "eeg"]))

    ax = sq.viz.plot_overview(rec, band=None, max_points=200, clip=False)
    spiked = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert spiked.max() > 1_000            # µV; survived the downsampling


def test_overview_clips_and_names_the_offender():
    rec, _ = sq.make_demo_recording()
    ax = sq.viz.plot_overview(rec, normalize=False)
    caption = " ".join(t.get_text() for t in ax.texts)
    assert "Fpz" in caption and "peak" in caption
    # Only the genuinely pathological channel — a caption naming most of the
    # montage would be noise, since stacked EEG traces normally overlap a little.
    assert caption.count("peak") <= 5


def test_overview_normalizes_mixed_channel_types():
    """EEG in µV and a DC channel in device units cannot share one scale: the
    larger type sets the spacing and every EEG trace becomes a hairline."""
    rec, _ = sq.make_demo_recording()
    assert len(set(str(t) for t in rec.ds.coords["ch_type"].values)) > 1

    auto = sq.viz.plot_overview(rec)                      # normalize="auto"
    assert "SD" in auto.get_ylabel()
    assert "per-channel SD" in auto.get_title()

    absolute = sq.viz.plot_overview(rec, normalize=False)
    assert "µV" in absolute.get_ylabel()


def test_overview_keeps_absolute_scale_for_a_single_type(scalp_rec):
    """With one channel type, µV are comparable between channels and should
    stay that way — normalising would throw away real information."""
    ax = sq.viz.plot_overview(scalp_rec)
    assert "µV" in ax.get_ylabel()
    assert "shared µV scale" in ax.get_title()


def test_uncovered_intervals_are_blank_not_zero(gapped_rec):
    """An interval inside a gap has no data; plotting 0% would read as total
    artifact, the opposite of 'nothing was recorded here'."""
    import numpy as np

    grid = sq.IntervalGrid.fixed(gapped_rec, 5.0)
    wmf = sq.compute([M.PeakToPeak()], gapped_rec, grid)
    ax = sq.viz.plot_clean_fraction(wmf, rec=gapped_rec, threshold=150.0)
    y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    assert np.isnan(y).any()
    assert np.nanmin(y) > 0


def test_snippet_plot_and_unknown_channel(scalp_rec):
    ax = sq.viz.plot_channel_snippet(scalp_rec, ["C3", "C4"], 0.0, 5.0)
    # Signal traces only — the 2-point scale bar is also a Line2D.
    traces = [ln for ln in ax.lines if len(ln.get_xdata()) > 2]
    assert len(traces) == 2
    with pytest.raises(KeyError):
        sq.viz.plot_channel_snippet(scalp_rec, ["NOPE"], 0.0, 5.0)


def test_snippet_picks_a_covered_window(gapped_rec):
    """The default window must not land inside the recording gap."""
    from signal_quality.viz.traces import _quiet_window

    t0 = _quiet_window(gapped_rec, 5.0)
    i0 = int(t0 * gapped_rec.sfreq)
    assert gapped_rec.covered[i0:i0 + int(5 * gapped_rec.sfreq)].all()
