"""Load a recording from a filesystem path.

``load(path)`` sniffs the path and dispatches to a reader. XLTEK studies and the
lossless HDF5 archive go through the vendored decoder, which preserves raw ADC
counts and the sample-stamp tables; everything else goes through MNE and loses
them (clipping detection and timestamp checks degrade gracefully).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.recording import Recording, build_dataset
from . import xltek


def load(path, line_freq: float = 60.0, verbose: bool = False, strict: bool = False) -> Recording:
    """Load any supported recording.

    Parameters
    ----------
    path : str | Path
        An XLTEK study directory (contains ``*.erd``), a lossless ``.h5``
        archive, or any file MNE can read (``.edf``, ``.vhdr``, ``.fif``, ...).
    line_freq : float
        Mains frequency at the recording site — 60 in the Americas, 50 elsewhere.
    strict : bool
        Passed to the XLTEK decoder: raise on shorted / mixed-rate channels
        instead of reporting them as defects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        if not list(path.glob("*.erd")):
            raise ValueError(f"{path} is a directory but contains no .erd file")
        return _from_xltek_dict(
            xltek.decode_study(path, verbose=verbose, strict=strict), path, line_freq
        )

    if path.suffix.lower() in (".h5", ".hdf5"):
        return _from_lossless_hdf5(path, line_freq)

    return _from_mne(path, line_freq)


def _from_xltek_dict(s, source_path, line_freq) -> Recording:
    """Map a ``decode_study()`` / HDF5 dict onto the canonical Dataset.

    Note this bypasses ``mne.io.RawArray`` entirely — the decoder already yields
    plain arrays, so there is no reason to round-trip through MNE's data model.
    Volts are recovered exactly as ``xltek._raw_from_decoded`` does it.
    """
    counts = s["counts"]
    signal = counts.astype(np.float64) * (s["factor_uV"][:, None] * 1e-6)

    ds = build_dataset(
        signal=signal,
        sfreq=s["sfreq"],
        ch_names=s["ch_names"],
        ch_types=s["ch_types"],
        ch_units=s.get("ch_unit"),
        counts=counts,
        factor_uV=s["factor_uV"],
        covered=s["covered"],
        meas_date=s.get("meas_date"),
        line_freq=line_freq,
    )

    gaps = xltek._gap_annots(s["covered"], s["sfreq"])
    ann = pd.DataFrame(
        list(gaps) + list(s.get("annotations", [])),
        columns=["onset", "duration", "description"],
    ).sort_values("onset", ignore_index=True)

    prov = dict(s.get("provenance", {}))
    if s.get("stamps") is not None:
        prov["stamps"] = s["stamps"]
    prov["first_stamp"] = s.get("first_stamp")

    return Recording(ds, Path(source_path), ann, prov, list(s.get("defects", [])))


def _from_lossless_hdf5(path, line_freq) -> Recording:
    import h5py

    with h5py.File(path, "r") as h:
        if h.attrs.get("format") != "xltek-lossless-hdf5":
            return _from_mne(path, line_freq)
        md = h.attrs.get("meas_date", "")
        s = dict(
            counts=h["counts"][()],
            factor_uV=h["factor_uV"][()],
            covered=h["covered"][()].astype(bool),
            sfreq=float(h.attrs["sfreq"]),
            ch_names=[x.decode() for x in h["ch_names"][()]],
            ch_types=[x.decode() for x in h["ch_types"][()]],
            ch_unit=([x.decode() for x in h["ch_unit"][()]] if "ch_unit" in h else None),
            meas_date=md or None,
            first_stamp=int(h.attrs.get("first_stamp", 0)),
            annotations=list(
                zip(
                    h["ann_onset"][()].tolist(),
                    h["ann_duration"][()].tolist(),
                    [x.decode() for x in h["ann_description"][()]],
                    strict=True,
                )
            ),
            provenance={k[5:]: v for k, v in h.attrs.items() if k.startswith("prov_")},
        )
    return _from_xltek_dict(s, path, line_freq)


def _from_mne(path, line_freq) -> Recording:
    """Anything MNE reads. No raw counts and no stamp tables are available, so
    clipping and timestamp checks report themselves as unavailable; coverage is
    reconstructed from ``BAD_`` annotations."""
    import mne

    raw = mne.io.read_raw(str(path), preload=True, verbose="error")
    sfreq = float(raw.info["sfreq"])
    n_times = raw.n_times

    covered = np.ones(n_times, dtype=bool)
    rows = []
    for a in raw.annotations:
        rows.append((float(a["onset"]), float(a["duration"]), str(a["description"])))
        if str(a["description"]).startswith("BAD_"):
            i0 = max(0, int(round(a["onset"] * sfreq)))
            i1 = min(n_times, int(round((a["onset"] + a["duration"]) * sfreq)))
            covered[i0:i1] = False

    ds = build_dataset(
        signal=raw.get_data(),
        sfreq=sfreq,
        ch_names=list(raw.ch_names),
        ch_types=raw.get_channel_types(),
        counts=None,
        covered=covered,
        meas_date=raw.info["meas_date"],
        line_freq=line_freq,
    )
    ann = pd.DataFrame(rows, columns=["onset", "duration", "description"])
    return Recording(ds, Path(path), ann, {"reader": "mne"}, [])
