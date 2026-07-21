"""Each metric must respond to its own fault and stay quiet on the others."""
from __future__ import annotations

import numpy as np
import pytest

import signal_quality as sq
from signal_quality import metrics as M


def _table(rec, metrics, grid=None):
    grid = grid or sq.IntervalGrid.whole(rec)
    return sq.compute(metrics, rec, grid).table


def test_rms_separates_loud_and_flat(faulty_rec):
    t = _table(faulty_rec, [M.RMS()])["rms"].droplevel("interval")
    assert t["FLAT"] == pytest.approx(0.0, abs=1e-9)
    assert t["LOUD"] > 5 * t["C3"]


def test_flat_fraction_finds_the_dead_channel(faulty_rec):
    t = _table(faulty_rec, [M.FlatFraction()])["flat_frac"].droplevel("interval")
    assert t["FLAT"] == pytest.approx(1.0)
    assert t.drop("FLAT").max() < 0.02


def test_line_ratio_finds_mains_pickup(faulty_rec):
    t = _table(faulty_rec, [M.LineRatio()])["line_ratio"].droplevel("interval")
    assert t["LINE"] > 100 * t["C3"]
    assert t.drop(["LINE", "FLAT"]).max() < 10


def test_line_ratio_follows_the_line_frequency(faulty_rec):
    """A 60 Hz tone must not be flagged when the site runs on 50 Hz."""
    at60 = _table(faulty_rec, [M.LineRatio(f0=60.0)])["line_ratio"]
    at50 = _table(faulty_rec, [M.LineRatio(f0=50.0)])["line_ratio"]
    assert at60.droplevel("interval")["LINE"] > 100
    assert at50.droplevel("interval")["LINE"] < 10


def test_max_corr_flags_bridge_and_isolation(faulty_rec):
    t = _table(faulty_rec, [M.MaxCorrelation()])["max_corr"].droplevel("interval")
    assert t["BRIDGE"] > 0.97          # near-copy of C3
    assert t["C3"] > 0.97              # ... which implicates C3 too
    assert t["FLAT"] < 0.6             # dead channel correlates with nothing


def test_correlation_pairs_names_the_bridged_pair(faulty_rec):
    pairs = M.correlation_pairs(faulty_rec, threshold=0.95)
    top = set(pairs.iloc[0][["channel_a", "channel_b"]])
    assert top == {"C3", "BRIDGE"}


def test_clip_fraction_needs_counts(clipping_rec, faulty_rec):
    t = _table(clipping_rec, [M.ClipFraction()])["clip_pct"].droplevel("interval")
    assert t["RAILED"] > 10
    assert t["C3"] == pytest.approx(0.0)

    # Without raw counts the metric must report unavailability, not guess.
    mf = sq.compute([M.ClipFraction()], faulty_rec, sq.IntervalGrid.whole(faulty_rec))
    assert mf.table["clip_pct"].isna().all()
    assert any("counts" in n for n in mf.notes)


def test_emg_fraction_responds_to_high_frequency_power(clean_rec):
    rng = np.random.default_rng(9)
    X = clean_rec.ds["signal"].values.copy()
    t = np.arange(X.shape[1]) / clean_rec.sfreq
    X[1] += 30e-6 * np.sin(2 * np.pi * 35 * t) * rng.standard_normal(X.shape[1])
    from signal_quality.core.recording import Recording, build_dataset
    rec = Recording(build_dataset(X, clean_rec.sfreq, clean_rec.ch_names,
                                  ["eeg"] * len(clean_rec.ch_names)))
    t_emg = _table(rec, [M.EMGFraction()])["emg_pct"].droplevel("interval")
    assert t_emg.iloc[1] > 5 * t_emg.iloc[0]
    assert t_emg.drop(t_emg.index[1]).max() < 5   # background EEG is not "muscle"


def test_metrics_exclude_gap_samples(gapped_rec):
    """RMS over the whole recording must ignore the zero-filled gap."""
    rms_masked = _table(gapped_rec, [M.RMS()])["rms"].droplevel("interval")["C3"]
    X = gapped_rec.ds["signal"].values
    import mne
    naive = mne.filter.filter_data(X * 1e6, gapped_rec.sfreq, 1.0, 45.0,
                                   verbose="error")[0].std()
    # The gap drags the naive figure down; masking must not suffer that.
    assert rms_masked > naive * 1.05
