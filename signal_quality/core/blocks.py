"""Block planning for chunked processing.

Computing metrics over a whole recording at once materialises several
full-length float64 copies (microvolts, then one per filter band), which at
1-second granularity dwarfs the source data. Processing in blocks bounds that
cost.

The correctness requirement is that chunking must not change any number. A
filter applied to an isolated block would ring at the block edges, so each block
is loaded with **padding** on both sides, filtered, and then only the interior —
the part unaffected by edge effects — is used. The padding is derived from the
actual FIR length rather than guessed: a lower high-pass produces a
proportionally longer filter, so any fixed constant would eventually be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How much recording each block covers, before padding.
DEFAULT_BLOCK_S = 60.0


@dataclass
class Block:
    """One unit of work: intervals to compute, and the span to load for them."""

    interval_ids: list
    i_start: int  # first sample of the intervals themselves
    i_stop: int
    view_start: int  # padded span actually loaded and filtered
    view_stop: int

    @property
    def n_view(self) -> int:
        return self.view_stop - self.view_start


def filter_pad_samples(sfreq: float, l_freq, h_freq) -> int:
    """Samples of padding needed for a filter to be edge-effect free.

    MNE builds a linear-phase FIR and applies it zero-phase, so influence
    extends about half the filter length either side; padding by that half
    length reproduces whole-recording filtering to machine precision.
    """
    if l_freq is None and h_freq is None:
        return 0
    from mne.filter import create_filter

    h = create_filter(None, sfreq, l_freq, h_freq, verbose="error")
    return int(len(np.atleast_1d(h)) // 2 + 1)


def plan_blocks(
    grid,
    n_times: int,
    pad: int,
    block_s: float = DEFAULT_BLOCK_S,
    sfreq: float = 1.0,
    min_view: int = 0,
) -> list[Block]:
    """Group the grid's intervals into padded blocks.

    Intervals are never split across blocks — each is computed from one
    contiguous, fully-padded view, so no metric ever sees a partial window.
    """
    per_block = max(1, int(round(block_s * sfreq / max(1, _median_len(grid)))))
    ids = list(grid.table.index)
    starts = grid.table["i_start"].to_numpy()
    stops = grid.table["i_stop"].to_numpy()

    blocks = []
    for k in range(0, len(ids), per_block):
        sel = slice(k, k + per_block)
        i0, i1 = int(starts[sel].min()), int(stops[sel].max())
        v0, v1 = max(0, i0 - pad), min(n_times, i1 + pad)
        # A view shorter than the filter would itself distort; widen it (the
        # recording may simply be short, in which case we take all of it).
        if min_view and (v1 - v0) < min_view:
            deficit = min_view - (v1 - v0)
            v0 = max(0, v0 - deficit // 2 - 1)
            v1 = min(n_times, v1 + deficit // 2 + 1)
        blocks.append(Block(list(np.asarray(ids)[sel]), i0, i1, v0, v1))
    return blocks


def _median_len(grid) -> int:
    lens = (grid.table["i_stop"] - grid.table["i_start"]).to_numpy()
    return int(np.median(lens)) if lens.size else 1
