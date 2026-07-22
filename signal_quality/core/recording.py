"""The in-memory signal model.

A :class:`Recording` is a thin wrapper around an ``xarray.Dataset``. The Dataset
is the single source of truth for signal values and per-channel metadata; MNE is
used only at the edges (reading files, array-level filtering, sensor geometry)
and never appears in ``metrics/`` or ``filters.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class Recording:
    """One loaded dataset.

    ``ds`` holds:

    ==========  ==================================================================
    data vars   ``signal (channel, time)`` in volts; ``counts (channel, time)``
                raw ADC integers when the source preserves them; ``covered
                (time,)`` bool, False inside recording gaps
    coords      ``channel``, ``time``; ``ch_type``, ``ch_unit``, ``factor_uV``
                as channel-coords
    attrs       ``sfreq``, ``meas_date``, ``line_freq``
    ==========  ==================================================================

    Annotations are ragged (onset/duration/description), so they live beside the
    Dataset as a DataFrame rather than inside it.
    """

    ds: xr.Dataset
    source_path: Path | None = None
    annotations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["onset", "duration", "description"])
    )
    provenance: dict = field(default_factory=dict)
    defects: list[dict] = field(default_factory=list)

    # -- convenience accessors ------------------------------------------------
    @property
    def sfreq(self) -> float:
        return float(self.ds.attrs["sfreq"])

    @property
    def line_freq(self) -> float:
        """Mains frequency in Hz. 60 in the Americas, 50 in most of the world."""
        return float(self.ds.attrs.get("line_freq", 60.0))

    @property
    def ch_names(self) -> list[str]:
        return [str(c) for c in self.ds.coords["channel"].values]

    @property
    def n_times(self) -> int:
        return self.ds.sizes["time"]

    @property
    def duration(self) -> float:
        """Length of the timeline in seconds, gaps included."""
        return self.n_times / self.sfreq

    @property
    def has_counts(self) -> bool:
        return "counts" in self.ds

    @property
    def covered(self) -> np.ndarray:
        """Bool mask over time; False where no data was recorded."""
        if "covered" in self.ds:
            return np.asarray(self.ds["covered"].values, dtype=bool)
        return np.ones(self.n_times, dtype=bool)

    def pick(self, ch_type: str | None = None, names=None) -> Recording:
        """Return a Recording restricted to some channels."""
        ds = self.ds
        if ch_type is not None:
            ds = ds.isel(channel=(ds.coords["ch_type"] == ch_type).values)
        if names is not None:
            existing = {str(c) for c in ds.coords["channel"].values}
            keep = [n for n in names if n in existing]
            ds = ds.sel(channel=keep)
        return Recording(ds, self.source_path, self.annotations, self.provenance, self.defects)

    def to_mne(self):
        """Build an ``mne.io.RawArray`` for MNE-shaped work (bipolar montages,
        epoching, ``compute_psd``).

        Deliberately *not* used by ``metrics/`` or ``filters.py`` — those operate
        on xarray and numpy alone, so the library does not depend on MNE's data
        model. Intended for notebook use alongside the metric pipeline.
        """
        import mne

        info = mne.create_info(
            self.ch_names,
            self.sfreq,
            [str(t) for t in self.ds.coords["ch_type"].values],
        )
        raw = mne.io.RawArray(self.ds["signal"].values, info, verbose="error")
        md = self.ds.attrs.get("meas_date")
        if md:
            raw.set_meas_date(pd.Timestamp(md).to_pydatetime())
        ann = self.annotations
        if len(ann):
            raw.set_annotations(
                mne.Annotations(
                    ann["onset"].to_numpy(),
                    ann["duration"].to_numpy(),
                    ann["description"].astype(str).tolist(),
                    orig_time=raw.info["meas_date"],
                )
            )
        return raw

    def __repr__(self) -> str:
        return (
            f"<Recording {len(self.ch_names)} ch @ {self.sfreq:g} Hz, "
            f"{self.duration / 60:.1f} min, "
            f"{100 * self.covered.mean():.0f}% covered"
            f"{', counts' if self.has_counts else ''}>"
        )


def build_dataset(
    signal,
    sfreq,
    ch_names,
    ch_types=None,
    ch_units=None,
    counts=None,
    factor_uV=None,
    covered=None,
    meas_date=None,
    line_freq=60.0,
) -> xr.Dataset:
    """Assemble the canonical Dataset from plain arrays.

    Single construction path, shared by every reader, so all sources produce an
    identically shaped Dataset.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (channel, time), got shape {signal.shape}")
    n_chan, n_times = signal.shape

    def _check_len(name, value, expected):
        if value is not None and len(value) != expected:
            raise ValueError(f"{name} has {len(value)} entries for {expected} channels")

    _check_len("ch_names", ch_names, n_chan)
    _check_len("ch_types", ch_types, n_chan)
    _check_len("ch_units", ch_units, n_chan)
    _check_len("factor_uV", factor_uV, n_chan)
    if counts is not None and np.asarray(counts).shape != signal.shape:
        raise ValueError(
            f"counts shape {np.asarray(counts).shape} does not match signal {signal.shape}"
        )
    if covered is not None and len(covered) != n_times:
        raise ValueError(f"covered has {len(covered)} samples for {n_times} time points")
    time = np.arange(n_times) / float(sfreq)

    data_vars = {"signal": (("channel", "time"), signal)}
    if counts is not None:
        data_vars["counts"] = (("channel", "time"), np.asarray(counts))
    if covered is not None:
        data_vars["covered"] = (("time",), np.asarray(covered, dtype=bool))

    coords = {
        "channel": list(ch_names),
        "time": time,
        "ch_type": ("channel", list(ch_types or ["misc"] * n_chan)),
        "ch_unit": ("channel", list(ch_units or ["V"] * n_chan)),
    }
    if factor_uV is not None:
        coords["factor_uV"] = ("channel", np.asarray(factor_uV, dtype=np.float64))

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["sfreq"] = float(sfreq)
    ds.attrs["line_freq"] = float(line_freq)
    ds.attrs["meas_date"] = "" if meas_date is None else str(meas_date)
    return ds
