"""Adapters that convert results into other tools' formats.

Kept separate from ``filters.py`` so the policy layer stays free of any
dependency on MNE — metrics and filters must work on plain tables, and a
convenience converter is not a reason to couple them.
"""
from __future__ import annotations

import pandas as pd


def to_annotations(segments: pd.DataFrame, prefix: str = "BAD_"):
    """Bad segments as MNE annotations, so they can drive downstream rejection.

    Annotations are channel-scoped via ``ch_names``, which MNE's
    ``reject_by_annotation`` honours — so a bad stretch on one electrode excludes
    only that electrode, not the whole epoch across the montage. That is the
    point of flagging per interval rather than per channel.
    """
    import mne

    if segments is None or not len(segments):
        return mne.Annotations([], [], [])
    return mne.Annotations(
        onset=segments["t_start"].to_numpy(),
        duration=segments["duration"].to_numpy(),
        description=[prefix + (r or "quality") for r in segments["reasons"]],
        ch_names=[(c,) for c in segments["channel"]],
    )
