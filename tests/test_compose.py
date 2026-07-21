"""Composition: several metrics over one grid must join without misalignment.

This is the property the whole design rests on — if the join silently reindexes
or pads, every downstream filter is judging the wrong rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import signal_quality as sq
from signal_quality import metrics as M


def test_join_preserves_row_index(faulty_rec):
    grid = sq.IntervalGrid.fixed(faulty_rec, 10.0)
    single = [M.RMS().compute(_ctx(faulty_rec, grid), grid),
              M.LineRatio().compute(_ctx(faulty_rec, grid), grid)]
    joined = sq.compute([M.RMS(), M.LineRatio()], faulty_rec, grid).table

    assert joined.index.equals(single[0].index)
    assert joined.index.equals(single[1].index)
    assert not joined[["rms", "line_ratio"]].isna().any().any()
    # Values must survive the join untouched.
    np.testing.assert_allclose(joined["rms"].to_numpy(),
                               single[0]["rms"].to_numpy())


def test_grid_shared_across_metrics(faulty_rec):
    grid = sq.IntervalGrid.fixed(faulty_rec, 5.0)
    mf = sq.compute([M.RMS(), M.FlatFraction(), M.MaxCorrelation()],
                    faulty_rec, grid)
    n_ch = len(faulty_rec.ch_names)
    assert len(mf.table) == n_ch * len(grid)
    assert mf.table.index.names == ["channel", "interval"]
    assert set(mf.metrics) == {"rms", "flat_frac", "max_corr"}


def test_long_view_round_trips(faulty_rec):
    mf = sq.compute([M.RMS(), M.LineRatio()], faulty_rec,
                    sq.IntervalGrid.fixed(faulty_rec, 10.0))
    long = mf.long()
    assert set(long.columns) == {"channel", "interval", "metric", "value"}
    assert len(long) == len(mf.table) * len(mf.metrics)
    back = long.pivot_table(index=["channel", "interval"], columns="metric",
                            values="value")
    np.testing.assert_allclose(
        back["rms"].to_numpy(), mf.table["rms"].to_numpy(), rtol=1e-12)


def test_interval_grids(faulty_rec):
    assert len(sq.IntervalGrid.whole(faulty_rec)) == 1
    g = sq.IntervalGrid.fixed(faulty_rec, 10.0)
    assert len(g) == 4                                   # 40 s / 10 s
    assert g.table["t_end"].iloc[-1] == pytest.approx(40.0)
    ov = sq.IntervalGrid.fixed(faulty_rec, 10.0, overlap=0.5)
    assert len(ov) > len(g)


def test_grid_records_coverage(gapped_rec):
    g = sq.IntervalGrid.fixed(gapped_rec, 5.0)
    cov = g.table["coverage"]
    assert cov.max() == pytest.approx(1.0)
    assert cov.min() == pytest.approx(0.0)               # the 15-25 s hole
    assert (cov < 1.0).sum() == 2                        # two 5 s windows in it


def _ctx(rec, grid):
    from signal_quality.core.context import MetricContext
    return MetricContext(rec, grid, ch_type="eeg")
