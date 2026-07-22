"""The shared time axis every metric is computed over.

Building the grid once, up front, is what makes composition work: two metrics
computed over the same grid produce tables with an identical row index, so
joining them is a plain concat with no alignment guesswork.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IntervalGrid:
    """A set of half-open time intervals over one recording.

    ``table`` has one row per interval, indexed by ``interval``:
    ``t_start``, ``t_end`` (seconds), ``i_start``, ``i_stop`` (sample indices),
    and ``coverage`` — the fraction of samples in the interval that were
    actually recorded. Metrics mask to covered samples; ``coverage`` tells you
    how much data a given number was computed from.
    """

    table: pd.DataFrame
    sfreq: float

    #: Default flagging resolution. Quality varies within a channel — an
    #: electrode that is clean for 25 minutes and unusable for 3 deserves a
    #: different answer per minute, not one label for the recording.
    DEFAULT_WINDOW = 1.0

    def __len__(self) -> int:
        return len(self.table)

    def __iter__(self):
        """Yield ``(interval_id, i_start, i_stop)`` triples."""
        for iid, row in self.table.iterrows():
            yield iid, int(row["i_start"]), int(row["i_stop"])

    @property
    def index(self) -> pd.Index:
        return self.table.index

    # -- constructors ---------------------------------------------------------
    @classmethod
    def whole(cls, rec) -> IntervalGrid:
        """One interval spanning the entire recording.

        Reproduces the whole-recording per-channel summary of the original
        notebook's quality table.
        """
        return cls._build(rec, [(0, rec.n_times)])

    @classmethod
    def fixed(cls, rec, duration: float, overlap: float = 0.0) -> IntervalGrid:
        """Fixed-length windows of ``duration`` seconds.

        Trailing samples that cannot fill a whole window are dropped, matching
        the original's ``reshape(-1, wlen)`` behaviour.
        """
        if duration <= 0:
            raise ValueError("duration must be positive")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        w = int(round(duration * rec.sfreq))
        step = max(1, int(round(w * (1 - overlap))))
        bounds = [(s, s + w) for s in range(0, rec.n_times - w + 1, step)]
        if not bounds:
            raise ValueError(
                f"window of {duration}s does not fit in a {rec.duration:.1f}s recording")
        return cls._build(rec, bounds)

    @classmethod
    def from_annotations(cls, rec, match: str, pad: float = 0.0) -> IntervalGrid:
        """One interval per annotation whose description contains ``match``."""
        ann = rec.annotations
        hit = ann[ann["description"].astype(str).str.contains(match, case=False)]
        if not len(hit):
            raise ValueError(f"no annotation matching {match!r}")
        bounds = []
        for _, a in hit.iterrows():
            i0 = int(round((a["onset"] - pad) * rec.sfreq))
            i1 = int(round((a["onset"] + a["duration"] + pad) * rec.sfreq))
            bounds.append((max(0, i0), min(rec.n_times, max(i1, i0 + 1))))
        return cls._build(rec, bounds)

    @classmethod
    def from_bounds(cls, rec, spans) -> IntervalGrid:
        """Explicit ``(t_start, t_end)`` pairs in seconds."""
        bounds = [(int(round(a * rec.sfreq)), int(round(b * rec.sfreq)))
                  for a, b in spans]
        return cls._build(rec, bounds)

    @classmethod
    def _build(cls, rec, bounds) -> IntervalGrid:
        cov = rec.covered
        rows = []
        for i0, i1 in bounds:
            seg = cov[i0:i1]
            rows.append(dict(
                t_start=i0 / rec.sfreq,
                t_end=i1 / rec.sfreq,
                i_start=i0,
                i_stop=i1,
                coverage=float(seg.mean()) if seg.size else 0.0,
            ))
        table = pd.DataFrame(rows)
        table.index = pd.RangeIndex(len(table), name="interval")
        return cls(table, rec.sfreq)
