#!/usr/bin/env python3
"""
Standalone reader for Natus / XLTEK NeuroWorks ("KTLX") EEG studies -> MNE-Python.

MNE-Python has no native XLTEK/.erd reader, so this module decodes the raw,
delta-compressed ``.erd`` file itself (pure Python + NumPy) and returns a normal
``mne.io.RawArray``.  From there you use plain MNE for everything: filtering,
plotting, spectra, EDF/FIF export, etc.  No third-party EEG library is required
to read the data; MNE is only used to *hold* it.

The format (reverse-engineered; schemas 8/9) for one study directory:
  <name>.erd  raw signal, delta-compressed          (decoded here)
  <name>.etc  table of contents: record offsets + sample stamps
  <name>.stc  segment table of contents: start/end sample stamps
  <name>.snc  sample <-> wall-clock (Windows FILETIME) sync table
  <name>.ent  notes: channel montage + clinical annotations

Works for the single-file layout produced by *XLTEK XChange* as well as the
classic multi-segment layout (<name>_000.erd, _001.erd, ...).

    from xltek import read_raw_xltek
    raw = read_raw_xltek("/path/to/STUDY_DIR")   # -> mne.io.RawArray
"""

from __future__ import annotations

import datetime as _dt
import re
import struct
from pathlib import Path

import numpy as np

# --- headbox conversion table (uV per raw unit), ported from the public -------
# --- wonambi/ktlx description; depends on the amplifier "headbox" model. ------
_EEG = lambda n, db: np.ones(n) * (8711.0 / (2**21 - 0.5)) * 2**db
_DC10800 = lambda n, db: np.ones(n) * ((10800000 / 65536) / (2**6)) * 2**db
_DC10000 = lambda n, db: np.ones(n) * ((10000000 / 65536) / (2**6)) * 2**db
_UNIT = lambda n, db: np.ones(n) * (1 / (2**6)) * 2**db
_OLD = lambda n, db: np.ones(n) * ((5000000.0 / (2**10 - 0.5)) / (2**6)) * 2**db


def _conversion(headbox_type, n_chan, db):
    ht = headbox_type[0]
    if ht in (1, 3, 19):
        f = _EEG(n_chan, db)
    elif ht == 4:
        f = np.concatenate([_EEG(24, db), _OLD(4, db)])
    elif ht == 6:
        f = np.concatenate([_EEG(32, db), _OLD(4, db)])
    elif ht == 8:
        f = np.concatenate([_EEG(25, db), _UNIT(2, db)])
    elif ht == 9:
        f = np.concatenate([_EEG(33, db), _UNIT(2, db)])
    elif ht == 14:
        f = np.concatenate([_EEG(38, db), _DC10800(10, db), _UNIT(2, db)])
    elif ht == 15:
        f = np.concatenate([_EEG(24, db), _EEG(4, db), _DC10000(4, db), _UNIT(2, db)])
    elif ht == 17:
        f = np.concatenate([_EEG(40, db), _DC10800(4, db), _UNIT(2, db)])
    elif ht == 21:
        f = np.concatenate([_EEG(128, db), _UNIT(2, db), _EEG(126, db)])
    elif ht == 22:
        f = np.concatenate([_EEG(32, db), _DC10800(8, db), _UNIT(2, db), _DC10800(1, db)])
    elif ht == 23:
        f = np.concatenate([_EEG(32, db), _DC10800(4, db), _UNIT(2, db), _DC10800(1, db)])
    else:
        raise NotImplementedError(f"conversion factor for headbox {ht} not known")
    return f[:n_chan]


def _cstr(b):
    return b.split(b"\x00", 1)[0].decode("utf-8", "replace")


def read_erd_header(erd_path: Path) -> dict:
    """Parse the fixed header of an .erd/.etc/.stc file (schema 8/9)."""
    with open(erd_path, "rb") as f:
        f.read(16)  # GUID
        schema, base = struct.unpack("<HH", f.read(4))
        if schema not in (7, 8, 9):
            raise NotImplementedError(f"file_schema {schema} not supported")
        (ctime,) = struct.unpack("<i", f.read(4))
        f.read(8)  # patient_id, study_id
        last = _cstr(f.read(80))
        first = _cstr(f.read(80))
        f.read(80)  # middle name
        pid = _cstr(f.read(80))
        assert f.tell() == 352
        (sfreq,) = struct.unpack("<d", f.read(8))
        (n_chan,) = struct.unpack("<i", f.read(4))
        (deltabits,) = struct.unpack("<i", f.read(4))
        phys_chan = struct.unpack(f"<{n_chan}i", f.read(4 * n_chan))
        f.seek(4464)
        headbox_type = struct.unpack("<4i", f.read(16))
        headbox_sn = struct.unpack("<4i", f.read(16))
        f.read(40 + 10 + 10)  # sw/hw version strings
        (discardbits,) = struct.unpack("<i", f.read(4))
        shorted = freq_factor = None
        if schema >= 8:
            shorted = struct.unpack("<1024h", f.read(2048))[:n_chan]
            freq_factor = struct.unpack("<1024h", f.read(2048))[:n_chan]
    return dict(
        file_schema=schema,
        creation_time=ctime,
        sfreq=sfreq,
        n_chan=n_chan,
        deltabits=deltabits,
        discardbits=discardbits,
        headbox_type=headbox_type,
        headbox_sn=headbox_sn,
        shorted=shorted,
        freq_factor=freq_factor,
        phys_chan=phys_chan,
        patient_id=pid,
        first_name=first,
        last_name=last,
    )


def read_etc(etc_path: Path) -> np.ndarray:
    """Table of contents: one record per compressed packet."""
    dt = np.dtype(
        [
            ("offset", "<i4"),
            ("samplestamp", "<i4"),
            ("sample_num", "<i4"),
            ("sample_span", "<i2"),
            ("unknown", "<i2"),
        ]
    )
    with open(etc_path, "rb") as f:
        f.seek(352)
        return np.fromfile(f, dtype=dt)


def read_stc(stc_path: Path):
    """Segment table of contents -> list of (segment_name, start, end)."""
    dt = np.dtype(
        [
            ("segment_name", "S256"),
            ("start_stamp", "<i4"),
            ("end_stamp", "<i4"),
            ("sample_num", "<i4"),
            ("sample_span", "<i4"),
        ]
    )
    with open(stc_path, "rb") as f:
        f.seek(352 + 4 + 4 + 48)  # skip general stc fields
        stamps = np.fromfile(f, dtype=dt)
    return stamps


def _decode_packet(f, offset, n_smp, n_chan, abs_delta=-1):
    """Decode one delta-compressed packet -> int array [n_chan, n_smp]."""
    l_mask = (n_chan + 7) // 8
    out = np.empty((n_chan, n_smp), dtype=np.int64)
    f.seek(offset)
    for i in range(n_smp):
        ev = f.read(1)
        if ev not in (b"\x00", b"\x01"):
            raise ValueError(f"bad event byte {ev!r} at sample {i}")
        mbytes = np.frombuffer(f.read(l_mask), dtype=np.uint8)
        bits = np.unpackbits(mbytes[::-1])[: -n_chan - 1 : -1].astype(bool)  # True=int16
        n_bytes = int(bits.sum()) + bits.size
        codes = np.where(bits, b"h", b"b")
        fmt = b"<" + codes.tobytes()
        rel = np.array(struct.unpack(fmt.decode(), f.read(n_bytes)))
        read_abs = bits & (rel == abs_delta)
        prev = out[:, i - 1] if i > 0 else np.zeros(n_chan, dtype=np.int64)
        out[~read_abs, i] = prev[~read_abs] + rel[~read_abs]
        if read_abs.any():
            absvals = np.frombuffer(f.read(4 * int(read_abs.sum())), dtype="<i4")
            out[read_abs, i] = absvals
    return out


def decode_study(study_dir, verbose=True, strict=False):
    """Decode a KTLX/XLTEK study to raw integer counts + full metadata.

    This is the lossless ground truth. Physical microvolts are recovered as
    ``counts.astype(float64) * factor_uV[:, None]``; Volts = uV * 1e-6.

    Returns a dict with keys: counts (int32 [n_chan, N], 0 in gaps), factor_uV
    (float64 [n_chan]), covered (bool [N]), sfreq, n_chan, ch_names, ch_types,
    ch_unit, meas_date (tz-aware UTC or None), first_stamp, annotations
    (list of (onset_s, duration_s, description) — clinical only), stamps
    (the raw .etc/.stc tables, for timestamp integrity checks), defects
    (list of dicts describing channel-alignment faults), provenance.

    Shorted channels and mixed per-channel sample rates are *reported* in
    ``defects`` rather than raised, so a faulty study can still be loaded and
    audited — reporting such faults is the point of this library. Pass
    ``strict=True`` to restore the original hard failure.
    """
    study = Path(study_dir)
    stems = list(study.glob("*.erd"))
    if not stems:
        raise FileNotFoundError(f"no .erd file in {study}")
    base = sorted(stems, key=lambda p: len(p.name))[0].with_suffix("")  # shortest = master

    hdr = read_erd_header(base.with_suffix(".erd"))
    n_chan, sfreq = hdr["n_chan"], hdr["sfreq"]
    defects = []
    if hdr["shorted"] and any(hdr["shorted"]):
        which = [i for i, s in enumerate(hdr["shorted"]) if s]
        if strict:
            raise NotImplementedError("shorted channels present; not handled")
        defects.append(
            dict(
                check="shorted_channels",
                channels=which,
                detail=f"{len(which)} channel(s) marked shorted in the header; "
                "their signal is not meaningful",
            )
        )
    if hdr["freq_factor"] and len(set(hdr["freq_factor"])) > 1:
        rates = sorted(set(hdr["freq_factor"]))
        if strict:
            raise NotImplementedError("mixed per-channel sample rates; not handled")
        defects.append(
            dict(
                check="mixed_sample_rates",
                channels=None,
                detail=f"per-channel freq_factor takes {len(rates)} distinct values "
                f"{rates}; channels cannot share one time axis",
            )
        )

    factor_uV = _conversion(hdr["headbox_type"], n_chan, hdr["discardbits"])

    # full sample timeline from the segment table + packet table
    stamps = read_stc(base.with_suffix(".stc"))
    etc = read_etc(base.with_suffix(".etc"))
    ss_all = etc["samplestamp"].astype("int64")
    span_all = etc["sample_span"].astype("int64")
    beg = int(stamps["start_stamp"].min())
    end = max(int(stamps["end_stamp"].max()) + 1, int((ss_all + span_all).max()))
    N = end - beg
    counts = np.zeros((n_chan, N), dtype=np.int32)
    covered = np.zeros(N, dtype=bool)

    if verbose:
        print(f"decoding {len(etc)} packets ({N} samples x {n_chan} ch)...")
    with open(base.with_suffix(".erd"), "rb") as f:
        for r in range(len(etc)):
            ss = int(ss_all[r])
            span = int(span_all[r])
            pkt = _decode_packet(f, int(etc["offset"][r]), span, n_chan)
            s0 = ss - beg
            w = min(span, N - s0)  # clip defensively
            counts[:, s0 : s0 + w] = pkt[:, :w]
            covered[s0 : s0 + w] = True

    named = _channel_names(base) or [f"ch{i:03d}" for i in range(n_chan)]
    ch_names = named[:n_chan]
    ch_types = [_ch_type(c) for c in ch_names]
    ch_unit = [_ch_unit(c) for c in ch_names]
    md = _snc_start_utc(base)
    meas_date = md.replace(tzinfo=_dt.UTC) if md is not None else None
    annotations = [(t, 0.0, name) for t, name in _annotations(base, sfreq, beg)]
    if len(named) != n_chan:
        defects.append(
            dict(
                check="channel_count_mismatch",
                channels=None,
                detail=f"header declares {n_chan} channels, montage names {len(named)}",
            )
        )
    return dict(
        counts=counts,
        factor_uV=factor_uV,
        covered=covered,
        sfreq=sfreq,
        n_chan=n_chan,
        ch_names=ch_names,
        ch_types=ch_types,
        ch_unit=ch_unit,
        meas_date=meas_date,
        first_stamp=beg,
        annotations=annotations,
        stamps=dict(etc=etc, stc=stamps),
        defects=defects,
        provenance=dict(
            file_schema=hdr["file_schema"],
            headbox_type=list(hdr["headbox_type"]),
            discardbits=hdr["discardbits"],
            patient_last=hdr["last_name"],
            patient_first=hdr["first_name"],
            patient_id=hdr["patient_id"],
            study=base.name,
        ),
    )


def _gap_annots(covered, sfreq):
    """Reconstruct BAD_gap annotations (onset, dur, desc) from the covered mask."""
    out, N = [], len(covered)
    edges = np.diff(covered.astype(np.int8))
    for st in np.where(edges == -1)[0]:
        e = np.where(edges[st:] == 1)[0]
        dur = (int(e[0]) if len(e) else N - 1 - st) / sfreq
        out.append(((st + 1) / sfreq, dur, "BAD_gap"))
    return out


def _raw_from_decoded(s):
    """Build an ``mne.io.RawArray`` from a decode_study() dict (single code path
    for both the .erd reader and the HDF5 reader, guaranteeing identical output)."""
    import mne

    data = s["counts"].astype(np.float64)
    data *= s["factor_uV"][:, None] * 1e-6  # counts -> uV -> V
    info = mne.create_info(list(s["ch_names"]), s["sfreq"], list(s["ch_types"]))
    raw = mne.io.RawArray(data, info, verbose="error")
    if s["meas_date"] is not None:
        raw.set_meas_date(s["meas_date"])
    anns = _gap_annots(s["covered"], s["sfreq"]) + list(s["annotations"])
    if anns:
        on, du, de = zip(*anns, strict=True)
        raw.set_annotations(mne.Annotations(on, du, de, orig_time=raw.info["meas_date"]))
    return raw


def read_raw_xltek(study_dir, verbose=True):
    """Read a KTLX/XLTEK study directory and return an ``mne.io.RawArray``
    (signal in Volts; gaps zero-filled + marked ``BAD_gap``; clinical notes as
    annotations)."""
    return _raw_from_decoded(decode_study(study_dir, verbose=verbose))


# ----------------------------------------------------------------- .ent / .snc
def _channel_names(base: Path):
    """Extract the ChanNames list from the .ent montage note."""
    for ext in (".ent", ".ent.old"):
        p = base.with_suffix(ext)
        if not p.exists():
            continue
        txt = p.read_bytes().decode("latin-1", "replace")
        i = txt.find("ChanNames")
        if i == -1:
            continue
        a = txt.find("(", i)
        b = txt.find(")", a)
        return [x.strip('" ') for x in txt[a + 1 : b].split(",") if x.strip('" ')]
    return None


def _snc_start_utc(base: Path):
    p = base.with_suffix(".snc")
    if not p.exists():
        return None
    b = p.read_bytes()
    stamp, lo, hi = struct.unpack("<iII", b[0x160 : 0x160 + 12])
    ft = (hi << 32) | lo
    return _dt.datetime(1601, 1, 1) + _dt.timedelta(microseconds=ft / 10)


def _annotations(base: Path, sfreq, beg):
    """Yield (onset_seconds, text) clinical notes parsed from the .ent file."""
    for ext in (".ent", ".ent.old"):
        p = base.with_suffix(ext)
        if not p.exists():
            continue
        data = p.read_bytes()
        pos, out = 352, []
        while pos + 16 <= len(data):
            typ, length, prev, unused = struct.unpack("<iiii", data[pos : pos + 16])
            if typ == 0 or length <= 16 or pos + length > len(data):
                break
            body = data[pos + 16 : pos + length].decode("latin-1", "replace")
            pos += length
            mtext = re.search(r'"Text",\s*"([^"]*)"', body)
            mstamp = re.search(r'"Stamp",\s*(\d+)', body)
            muser = re.search(r'"User",\s*"([^"]*)"', body)
            if not mtext or not mstamp or not mtext.group(1):
                continue
            if mtext.group(1) == "Analyzed Data Note":
                continue
            user = muser.group(1).split()[0] if muser and muser.group(1) else "?"
            t = (int(mstamp.group(1)) - beg) / sfreq
            out.append((t, f"{mtext.group(1)} ({user})"))
        return out
    return []


def _ch_type(name):
    n = name.upper()
    if n.startswith("ECG") or n.startswith("EKG"):
        return "ecg"
    if n in {"CHEST", "ABD", "FLOW", "SNORE"}:
        return "resp"
    if n in {"OSAT", "PR", "POS"} or n.startswith("DC") or n.startswith("DIF"):
        return "misc"
    return "eeg"


def _ch_unit(name):
    """Physical unit label. OSAT/PR are device values digitized with unit gain
    (factor == 1), so they are stored as raw counts, not microvolts."""
    return "count" if name.upper() in {"OSAT", "PR"} else "uV"


# ============================================================ lossless exports
def export_hdf5(study_dir, out_path):
    """Bit-exact archive: raw integer counts + per-channel factors + all
    metadata, in a self-describing HDF5 file. Read back with read_raw_hdf5().
    Nothing is quantized or resampled; recovers the decoder output exactly."""
    import h5py

    s = decode_study(study_dir)
    with h5py.File(out_path, "w") as h:
        h.attrs["format"] = "xltek-lossless-hdf5"
        h.attrs["version"] = "1"
        h.attrs["sfreq"] = float(s["sfreq"])
        h.attrs["first_stamp"] = int(s["first_stamp"])
        h.attrs["meas_date"] = s["meas_date"].isoformat() if s["meas_date"] else ""
        h.attrs["README"] = (
            "Lossless archive of a Natus/XLTEK NeuroWorks EEG study. "
            "microvolts = counts.astype(float64) * factor_uV[:, None]; "
            "volts = microvolts * 1e-6. Samples where covered==0 are recording "
            "gaps (zero-filled). Load with xltek.read_raw_hdf5()."
        )
        for k, v in s["provenance"].items():
            h.attrs["prov_" + k] = str(v)
        h.create_dataset(
            "counts",
            data=s["counts"],
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=True,
        )
        h.create_dataset("factor_uV", data=s["factor_uV"])  # float64
        h.create_dataset("covered", data=s["covered"].astype("u1"), compression="gzip")
        h.create_dataset("ch_names", data=np.array(s["ch_names"], dtype="S32"))
        h.create_dataset("ch_types", data=np.array(s["ch_types"], dtype="S16"))
        h.create_dataset("ch_unit", data=np.array(s["ch_unit"], dtype="S16"))
        on = [a[0] for a in s["annotations"]]
        du = [a[1] for a in s["annotations"]]
        de = [a[2] for a in s["annotations"]]
        h.create_dataset("ann_onset", data=np.array(on, dtype="f8"))
        h.create_dataset("ann_duration", data=np.array(du, dtype="f8"))
        h.create_dataset("ann_description", data=np.array(de, dtype="S256"))
    return out_path


def read_raw_hdf5(path):
    """Load the bit-exact HDF5 archive back into an ``mne.io.RawArray`` (identical
    to read_raw_xltek() on the original study)."""
    import h5py

    with h5py.File(path, "r") as h:
        md = h.attrs.get("meas_date", "")
        s = dict(
            counts=h["counts"][()],
            factor_uV=h["factor_uV"][()],
            covered=h["covered"][()].astype(bool),
            sfreq=float(h.attrs["sfreq"]),
            ch_names=[x.decode() for x in h["ch_names"][()]],
            ch_types=[x.decode() for x in h["ch_types"][()]],
            meas_date=_dt.datetime.fromisoformat(md) if md else None,
            annotations=list(
                zip(
                    h["ann_onset"][()].tolist(),
                    h["ann_duration"][()].tolist(),
                    [x.decode() for x in h["ann_description"][()]],
                    strict=True,
                )
            ),
        )
    return _raw_from_decoded(s)


def export_brainvision(study_or_raw, out_vhdr):
    """Generic BrainVision export (float32 .vhdr/.vmrk/.eeg) readable by MNE,
    EEGLAB, FieldTrip, BrainVision Analyzer. float32 error is far below the
    amplifier's own resolution. Commas in note text are replaced with ';' so
    the .vmrk marker format (comma-delimited) stays valid."""
    import mne

    raw = study_or_raw if hasattr(study_or_raw, "info") else read_raw_xltek(study_or_raw)
    raw = raw.copy()
    if raw.annotations is not None and len(raw.annotations):
        desc = [d.replace(",", ";") for d in raw.annotations.description]
        raw.set_annotations(
            mne.Annotations(
                raw.annotations.onset,
                raw.annotations.duration,
                desc,
                orig_time=raw.info["meas_date"],
            )
        )
    mne.export.export_raw(str(out_vhdr), raw, fmt="brainvision", overwrite=True)
    return out_vhdr


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "."
    r = read_raw_xltek(d)
    print(r)
    print("meas_date:", r.info["meas_date"])
    print("channels :", r.ch_names)
    print("annotations:", len(r.annotations))
