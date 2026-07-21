"""The data model must stay independent of MNE.

MNE is allowed in exactly three places: reading files (``io/``), array-level
filtering (``core/context.py``), and standard electrode geometry
(``montage.py``). If it leaks into ``metrics/`` or ``filters.py``, the library
quietly becomes EEG-only and the xarray model stops earning its keep — so this
is enforced rather than merely documented.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "signal_quality"

#: Modules permitted to import MNE, and why.
ALLOWED = {
    "io/load.py": "reads formats MNE supports",
    "io/xltek.py": "vendored reader; builds RawArray for its own public API",
    "core/context.py": "mne.filter.filter_data is an array function",
    "core/recording.py": "to_mne() adapter, for notebook use",
    "montage.py": "standard 10-20 electrode coordinates",
    "viz/topomap.py": "mne.viz.plot_topomap, called with a plain xy array",
}


def _imports_mne(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "mne" for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "mne":
                return True
    return False


@pytest.mark.parametrize("path", sorted(PKG.rglob("*.py")), ids=lambda p: p.name)
def test_mne_only_at_the_edges(path):
    rel = path.relative_to(PKG).as_posix()
    if _imports_mne(path) and rel not in ALLOWED:
        pytest.fail(
            f"{rel} imports MNE but is not an edge module. Metrics and filters "
            f"must work on xarray/numpy alone; add an adapter or extend ALLOWED "
            f"with a justification.")


def test_metrics_and_filters_are_mne_free():
    """Stated explicitly, so the intent survives changes to ALLOWED."""
    offenders = [p.relative_to(PKG).as_posix()
                 for p in list((PKG / "metrics").rglob("*.py")) + [PKG / "filters.py"]
                 if _imports_mne(p)]
    assert offenders == []


def test_recording_survives_without_mne_imported():
    """Building and querying a Recording must not require MNE at all."""
    import numpy as np

    from signal_quality.core.recording import Recording, build_dataset

    ds = build_dataset(np.zeros((2, 100)), 100.0, ["a", "b"], ["eeg", "eeg"])
    rec = Recording(ds)
    assert rec.sfreq == 100.0
    assert rec.ch_names == ["a", "b"]
    assert rec.covered.all()
