"""Recording-scope integrity checks.

These are not per-channel measurements, so they do not join into the metric
table — a gap is a property of the recording, and broadcasting it across thirty
identical channel rows would be noise. They emit their own findings table:

    check | severity | t_start | t_end | channel | detail

The checks here are signal-agnostic: they ask whether the data exists, whether
its clock makes sense, and whether its channels can be put on a common time
axis, none of which depends on what was being measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["check", "severity", "t_start", "t_end", "channel", "detail"]


def _findings(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def coverage_gaps(rec, min_duration: float = 0.0) -> pd.DataFrame:
    """Stretches of the timeline where no data was recorded.

    Gaps are real missing time, not zeros to be analysed: every metric in this
    library masks them out, and a gap large enough to matter is itself a
    finding. Severity escalates once the gap exceeds 1% of the recording.
    """
    from ..io.xltek import _gap_annots

    cov = rec.covered
    rows = []
    for onset, dur, _ in _gap_annots(cov, rec.sfreq):
        if dur < min_duration:
            continue
        frac = dur / rec.duration
        rows.append(dict(
            check="data_gap",
            severity="bad" if frac > 0.01 else "marginal",
            t_start=onset, t_end=onset + dur, channel=None,
            detail=f"{dur:.1f} s with no data ({100 * frac:.1f}% of the recording)",
        ))
    missing = 1.0 - cov.mean()
    if missing > 0:
        rows.append(dict(
            check="coverage", severity="bad" if missing > 0.05 else "marginal",
            t_start=0.0, t_end=rec.duration, channel=None,
            detail=f"{100 * missing:.1f}% of the timeline has no data",
        ))
    return _findings(rows)


def timestamp_anomalies(rec, jitter_tol: float = 1e-6) -> pd.DataFrame:
    """Faults in the recording's own clock.

    A uniformly-sampled timeline has no per-sample clock to check, so the
    evidence lives in the acquisition system's sample-stamp tables, which the
    XLTEK reader preserves. Checks: stamps that go backwards, packets that
    overlap, and packets whose stamp spacing disagrees with their declared
    length (irregular sample periods). For sources with an explicit non-uniform
    time coordinate, the coordinate itself is checked instead.
    """
    rows = []
    stamps = (rec.provenance or {}).get("stamps")

    if stamps is None:
        t = rec.ds.coords["time"].values
        dt = np.diff(t)
        if dt.size:
            if (dt <= 0).any():
                n = int((dt <= 0).sum())
                i = int(np.argmax(dt <= 0))
                rows.append(dict(
                    check="nonmonotonic_time", severity="bad",
                    t_start=float(t[i]), t_end=float(t[min(i + 1, len(t) - 1)]),
                    channel=None,
                    detail=f"time coordinate decreases at {n} sample(s)"))
            spread = float(dt.max() - dt.min())
            if spread > jitter_tol:
                rows.append(dict(
                    check="irregular_sampling", severity="marginal",
                    t_start=float(t[0]), t_end=float(t[-1]), channel=None,
                    detail=f"sample period varies by {spread:.3e} s "
                           f"(min {dt.min():.6f}, max {dt.max():.6f})"))
        if not rows:
            rows.append(dict(
                check="timestamp_source", severity="info",
                t_start=0.0, t_end=rec.duration, channel=None,
                detail="no acquisition stamp table available; only the uniform "
                       "time coordinate could be checked"))
        return _findings(rows)

    etc = np.asarray(stamps["etc"])
    # Sample stamps are absolute acquisition counts; the recording timeline
    # starts at first_stamp. Without this offset every reported time is shifted
    # by the acquisition's start offset.
    beg = int((rec.provenance or {}).get("first_stamp") or 0)
    ss = etc["samplestamp"].astype("int64") - beg
    span = etc["sample_span"].astype("int64")
    sf = rec.sfreq
    cov = rec.covered

    back = np.where(np.diff(ss) < 0)[0]
    if back.size:
        rows.append(dict(
            check="nonmonotonic_time", severity="bad",
            t_start=float(ss[back[0]] / sf), t_end=float(ss[back[-1] + 1] / sf),
            channel=None,
            detail=f"{back.size} packet(s) have a sample stamp earlier than "
                   f"their predecessor"))

    step = np.diff(ss)
    declared = span[:-1]
    overlap = np.where(step < declared)[0]
    if overlap.size:
        rows.append(dict(
            check="overlapping_packets", severity="bad",
            t_start=float(ss[overlap[0]] / sf),
            t_end=float((ss[overlap[-1]] + span[overlap[-1]]) / sf), channel=None,
            detail=f"{overlap.size} packet(s) start before the previous packet ends"))

    # A stamp jump larger than the declared span is only a *clock* fault if it
    # is not simply the recording gap that coverage_gaps already reports.
    # Otherwise every gapped recording would also be accused of irregular
    # sampling, for the same underlying event.
    irregular = [i for i in np.where(step > declared)[0]
                 if not _is_known_gap(cov, ss[i] + span[i], ss[i + 1])]
    if irregular:
        gaps = (step - declared)
        worst = int(max(irregular, key=lambda i: gaps[i]))
        rows.append(dict(
            check="irregular_sampling", severity="marginal",
            t_start=float(ss[worst] / sf),
            t_end=float(ss[worst + 1] / sf), channel=None,
            detail=f"{len(irregular)} packet boundary/ies have a stamp gap larger "
                   f"than the declared span, unexplained by a recording gap "
                   f"(largest {gaps[worst] / sf:.3f} s)"))

    stc = stamps.get("stc")
    if stc is not None and len(stc):
        stc = np.asarray(stc)
        # Segment stamps are absolute too, so the same offset applies.
        seg_end = int(stc["end_stamp"].max()) - beg
        pkt_end = int((ss + span).max())
        if abs(seg_end + 1 - pkt_end) > 1:
            rows.append(dict(
                check="segment_inconsistent", severity="marginal",
                t_start=float(min(seg_end, pkt_end) / sf),
                t_end=float(max(seg_end, pkt_end) / sf), channel=None,
                detail=f"segment table ends at sample {seg_end + 1}, packet table "
                       f"at {pkt_end}"))

    return _findings(rows)


def _is_known_gap(cov, i0, i1, tol: float = 0.9) -> bool:
    """True if samples ``[i0, i1)`` are already known to be uncovered.

    Used to avoid reporting a recording gap a second time as a clock anomaly.
    """
    i0, i1 = int(max(0, i0)), int(min(len(cov), i1))
    if i1 <= i0:
        return False
    return bool((~cov[i0:i1]).mean() >= tol)


def channel_alignment(rec) -> pd.DataFrame:
    """Whether all channels can honestly share one time axis.

    Mixed per-channel sample rates and shorted channels are reported by the
    reader as defects (it used to refuse to open such studies outright); this
    surfaces them, and adds checks that depend on the assembled data.
    """
    rows = []
    for d in rec.defects or []:
        chans = d.get("channels")
        rows.append(dict(
            check=d.get("check", "reader_defect"), severity="bad",
            t_start=0.0, t_end=rec.duration,
            channel=(",".join(map(str, chans)) if chans else None),
            detail=d.get("detail", "")))

    # Channels that are constant for the whole recording carry no data at all,
    # as distinct from a channel that is merely flat some of the time.
    sig = rec.ds["signal"].values
    if sig.shape[1] > 1:
        dead = np.ptp(sig, axis=1) == 0
        for name in np.asarray(rec.ch_names)[dead]:
            rows.append(dict(
                check="dead_channel", severity="bad",
                t_start=0.0, t_end=rec.duration, channel=str(name),
                detail="channel is exactly constant for the whole recording; "
                       "no data was captured on it"))

    n_named = len(rec.ch_names)
    n_sig = sig.shape[0]
    if n_named != n_sig:
        rows.append(dict(
            check="channel_count_mismatch", severity="bad",
            t_start=0.0, t_end=rec.duration, channel=None,
            detail=f"{n_named} channel names for {n_sig} signal rows"))

    return _findings(rows)


def check_integrity(rec, checks=None) -> pd.DataFrame:
    """Run all recording-scope checks and concatenate the findings.

    Returns an empty (correctly-columned) frame when nothing is wrong.
    """
    checks = checks or (coverage_gaps, timestamp_anomalies, channel_alignment)
    out = [c(rec) for c in checks]
    out = [f for f in out if len(f)]
    if not out:
        return _findings([])
    return pd.concat(out, ignore_index=True).sort_values(
        "t_start", ignore_index=True)
