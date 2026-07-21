"""Electrode placement, including the legacy-label translation."""
from __future__ import annotations

import numpy as np

from signal_quality.montage import place, to_clinical, to_modern


def test_legacy_labels_are_placed_under_their_original_names():
    """T3/T4/T5/T6 are still used clinically but absent from standard montages.

    They must be found *and* handed back under the original label, or they stop
    matching the metric table.
    """
    placed, xy, unplaced = place(["T3", "T4", "T5", "T6", "Cz"])
    assert set(placed) == {"T3", "T4", "T5", "T6", "Cz"}
    assert xy.shape == (5, 2)
    assert np.isfinite(xy).all()
    assert unplaced == []


def test_nonstandard_electrodes_are_reported_not_dropped():
    """Silently discarding unplaceable channels would hide bad electrodes — in
    the reference study they were among the worst."""
    placed, xy, unplaced = place(["Fp1", "Cz", "T1", "T2", "OSAT"])
    assert set(unplaced) == {"T1", "T2", "OSAT"}
    assert set(placed) == {"Fp1", "Cz"}
    assert len(xy) == len(placed)


def test_positions_are_distinct_and_plain_numpy():
    placed, xy, _ = place(["Fp1", "Fp2", "O1", "O2", "Cz"])
    assert isinstance(xy, np.ndarray) and xy.dtype == float
    assert len({tuple(p) for p in xy}) == len(placed)


def test_label_translation_round_trips():
    assert to_modern("T3") == "T7"
    assert to_clinical("T7") == "T3"
    assert to_modern("Cz") == "Cz"           # identity where no rename applies
    assert to_clinical(to_modern("T5")) == "T5"


def test_all_unplaceable_is_not_a_crash():
    placed, xy, unplaced = place(["ECG", "OSAT", "PR"])
    assert placed == []
    assert xy.shape == (0, 2)
    assert len(unplaced) == 3
