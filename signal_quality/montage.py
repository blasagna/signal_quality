"""Scalp electrode positions for head-model plots.

The reference project repeated the same rename-and-place dance in four separate
places. It lives here once.

This is the only module outside ``io/`` that touches MNE, and it does so purely
to look up standard 10-20 coordinates: it takes channel names and returns a
plain ``(n, 2)`` array of xy positions, so nothing MNE-shaped reaches ``viz/``.
"""
from __future__ import annotations

import numpy as np

#: Superseded 10-20 labels still used clinically -> the modern names that the
#: standard montages actually contain.
LEGACY_RENAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_INVERSE = {v: k for k, v in LEGACY_RENAMES.items()}


def place(ch_names, montage: str = "standard_1020", sfreq: float = 1000.0):
    """Find 2-D scalp positions for as many channels as possible.

    Returns ``(placed, xy, unplaced)``:

    * ``placed`` — the channel names that have a position, under their
      *original* labels, so they still match the metric table.
    * ``xy`` — ``(len(placed), 2)`` topomap coordinates.
    * ``unplaced`` — names with no standard position (extended or non-scalp
      electrodes). Worth reporting rather than silently dropping: in the
      reference study the unplaced electrodes were among the worst-quality ones.
    """
    import mne

    names = [str(c) for c in ch_names]
    renamed = [LEGACY_RENAMES.get(n, n) for n in names]

    info = mne.create_info(renamed, sfreq, "eeg")
    with mne.utils.use_log_level("error"):
        info.set_montage(montage, match_case=False, on_missing="ignore")

    mont = info.get_montage()
    pos = mont.get_positions()["ch_pos"] if mont is not None else {}
    keep = [i for i, n in enumerate(renamed)
            if n in pos and np.isfinite(list(pos[n])).all()]

    if not keep:
        return [], np.zeros((0, 2)), names

    try:
        sub = mne.pick_info(info, np.asarray(keep))
        from mne.channels.layout import _find_topomap_coords
        xy = _find_topomap_coords(sub, picks=list(range(len(keep))))
    except Exception:
        # Fall back to a plain top-down projection of the 3-D coordinates.
        xy = np.array([[pos[renamed[i]][0], pos[renamed[i]][1]] for i in keep])
        xy = xy / (np.abs(xy).max() or 1) * 0.09

    placed = [names[i] for i in keep]
    unplaced = [n for i, n in enumerate(names) if i not in set(keep)]
    return placed, np.asarray(xy, dtype=float), unplaced


def to_modern(name: str) -> str:
    """``T3`` -> ``T7``. Identity for names that need no translation."""
    return LEGACY_RENAMES.get(name, name)


def to_clinical(name: str) -> str:
    """``T7`` -> ``T3``. Inverse of :func:`to_modern`."""
    return _INVERSE.get(name, name)
