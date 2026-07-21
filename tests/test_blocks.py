"""Chunked processing must not change any number.

This is the test that protects the block redesign. Blocks exist to bound memory;
if they also shifted results, every metric would silently depend on a tuning
parameter that has nothing to do with the signal.
"""
from __future__ import annotations

import numpy as np
import pytest

import signal_quality as sq
from signal_quality import metrics as M
from signal_quality.core.blocks import filter_pad_samples, plan_blocks

ALL_METRICS = lambda: [M.RMS(), M.PeakToPeak(), M.FlatFraction(),
                       M.MaxCorrelation(), M.LineRatio(), M.EMGFraction(),
                       M.ClipFraction()]


@pytest.fixture(scope="module")
def demo():
    return sq.make_demo_recording(duration=90.0)[0]


@pytest.mark.parametrize("block_s", [5.0, 13.0, 30.0])
def test_block_size_does_not_change_results(demo, block_s):
    grid = sq.IntervalGrid.fixed(demo, 1.0)
    ref = sq.compute(ALL_METRICS(), demo, grid, block_s=1e9).table
    got = sq.compute(ALL_METRICS(), demo, grid, block_s=block_s).table

    assert got.index.equals(ref.index)
    for col in ref.columns:
        a, b = ref[col].to_numpy(), got[col].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(b[both], a[both], rtol=1e-9, atol=1e-9,
                                   err_msg=f"{col} changed with block size")
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


def test_a_lower_highpass_needs_more_padding(demo):
    """Padding is derived from the actual FIR, not a constant: a 0.1 Hz
    high-pass needs an order of magnitude more context than a 1 Hz one."""
    sf = demo.sfreq
    assert filter_pad_samples(sf, 1.0, 45.0) < filter_pad_samples(sf, 0.1, 45.0)

    metric = M.RMS(l_freq=0.1, h_freq=45.0)
    grid = sq.IntervalGrid.fixed(demo, 1.0)
    ref = sq.compute([metric], demo, grid, block_s=1e9).table["rms"]
    got = sq.compute([metric], demo, grid, block_s=10.0).table["rms"]
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-8,
                               atol=1e-8)


def test_analysis_window_padding_is_additive(demo):
    """The widened spectral window and the filter's own context add up.

    Taking the maximum instead leaves the outermost analysis samples filtered
    against the block edge, which showed up as emg_pct differing from
    whole-recording values on 29% of intervals.
    """
    sf = demo.sfreq
    emg = M.EMGFraction(min_analysis_s=4.0)
    filt_pad = filter_pad_samples(sf, *emg.analysis_band()) / sf
    assert emg.required_pad_s(sf) == pytest.approx(filt_pad + 2.0)
    assert emg.required_pad_s(sf) > max(filt_pad, 2.0)


def test_metrics_only_see_samples_inside_the_loaded_view(demo):
    """A metric reaching outside its block must fail loudly, not read whatever
    happens to be adjacent in memory."""
    from signal_quality.core.context import MetricContext

    ctx = MetricContext(demo, None)
    ctx.set_view(1000, 2000)
    ctx.segment(1200, 1300)                      # inside: fine
    with pytest.raises(ValueError, match="outside the loaded view"):
        ctx.segment(500, 600)


def test_view_change_drops_cached_arrays(demo):
    from signal_quality.core.context import MetricContext

    ctx = MetricContext(demo, None)
    ctx.set_view(0, 5000)
    ctx.filtered(1.0, 45.0)
    assert ctx._cache
    ctx.set_view(5000, 10000)
    assert not ctx._cache, "stale block buffers must not survive a view change"


def test_blocks_cover_every_interval_exactly_once(demo):
    grid = sq.IntervalGrid.fixed(demo, 1.0)
    blocks = plan_blocks(grid, demo.n_times, pad=100, block_s=7.0,
                         sfreq=demo.sfreq)
    seen = [i for b in blocks for i in b.interval_ids]
    assert sorted(seen) == sorted(grid.table.index)
    assert len(seen) == len(set(seen))
    for b in blocks:
        assert b.view_start <= b.i_start and b.view_stop >= b.i_stop


def test_default_grid_is_one_second(demo):
    mf = sq.compute([M.RMS()], demo)
    widths = mf.table["t_end"] - mf.table["t_start"]
    assert np.allclose(widths.to_numpy(), 1.0)
    assert sq.IntervalGrid.DEFAULT_WINDOW == 1.0
