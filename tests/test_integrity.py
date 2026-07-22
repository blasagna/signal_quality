"""Generic integrity checks: data existence, clock sanity, channel alignment."""

from __future__ import annotations

import numpy as np
import pytest

import signal_quality as sq
from signal_quality.core.recording import Recording
from signal_quality.metrics import integrity


def test_gap_is_found_with_correct_extent(gapped_rec):
    issues = integrity.coverage_gaps(gapped_rec)
    gaps = issues[issues["check"] == "data_gap"]
    assert len(gaps) == 1
    assert gaps["t_start"].iloc[0] == pytest.approx(15.0, abs=0.05)
    assert (gaps["t_end"] - gaps["t_start"]).iloc[0] == pytest.approx(10.0, abs=0.05)
    assert gaps["severity"].iloc[0] == "bad"


def test_clean_recording_reports_no_gaps(clean_rec):
    assert len(integrity.coverage_gaps(clean_rec)) == 0


def test_nonmonotonic_stamps_are_caught(gapped_rec, stamp_tables):
    rec = _with_stamps(gapped_rec, stamp_tables["backwards"])
    checks = set(integrity.timestamp_anomalies(rec)["check"])
    assert "nonmonotonic_time" in checks


def test_irregular_sample_period_is_caught(gapped_rec, stamp_tables):
    rec = _with_stamps(gapped_rec, stamp_tables["jumpy"])
    out = integrity.timestamp_anomalies(rec)
    assert "irregular_sampling" in set(out["check"])


def test_overlapping_packets_are_caught(gapped_rec, stamp_tables):
    rec = _with_stamps(gapped_rec, stamp_tables["overlapping"])
    assert "overlapping_packets" in set(integrity.timestamp_anomalies(rec)["check"])


def test_stamp_times_are_offset_by_first_stamp(gapped_rec, stamp_tables):
    """Stamps are absolute acquisition counts; findings must be reported on the
    recording's own timeline, not shifted by where acquisition started."""
    etc = stamp_tables["backwards"].copy()
    offset = 474136
    etc["samplestamp"] += offset
    rec = _with_stamps(gapped_rec, etc, first_stamp=offset)

    out = integrity.timestamp_anomalies(rec)
    bad = out[out["check"] == "nonmonotonic_time"]
    assert len(bad)
    # Without the offset these land ~1850 s out, far past the end of the data.
    assert 0 <= bad["t_start"].iloc[0] <= rec.duration


def test_recording_gap_is_not_double_reported_as_a_clock_fault(gapped_rec, stamp_tables):
    """A stamp jump that coincides with a known gap is the gap, not a second,
    independent clock anomaly."""
    sf = gapped_rec.sfreq
    span = 250
    # Packets run contiguously up to the fixture's hole at 15 s, resume at 25 s.
    before = np.arange(0, int(15 * sf), span)
    after = np.arange(int(25 * sf), int(35 * sf), span)
    etc = np.zeros(len(before) + len(after), dtype=stamp_tables["ok"].dtype)
    etc["samplestamp"] = np.concatenate([before, after])
    etc["sample_span"] = span
    rec = _with_stamps(gapped_rec, etc)

    out = integrity.timestamp_anomalies(rec)
    assert "irregular_sampling" not in set(out["check"])

    # Sanity: the same jump *not* backed by a gap must still be reported, so the
    # test is proving suppression rather than a dead check.
    shifted = etc.copy()
    shifted["samplestamp"][len(before) :] += int(3 * sf)
    assert "irregular_sampling" in set(
        integrity.timestamp_anomalies(_with_stamps(gapped_rec, shifted))["check"]
    )


def test_well_formed_stamps_are_quiet(gapped_rec, stamp_tables):
    rec = _with_stamps(gapped_rec, stamp_tables["ok"])
    out = integrity.timestamp_anomalies(rec)
    assert not len(out[out["severity"].isin(["bad", "marginal"])])


def test_missing_stamp_table_is_reported_as_unknown(clean_rec):
    """Absence of evidence must be labelled, not reported as a clean bill."""
    out = integrity.timestamp_anomalies(clean_rec)
    assert set(out["check"]) == {"timestamp_source"}
    assert out["severity"].iloc[0] == "info"


def test_reader_defects_surface_as_alignment_findings(clean_rec):
    clean_rec.defects = [
        dict(check="mixed_sample_rates", channels=None, detail="two distinct rates")
    ]
    out = integrity.channel_alignment(clean_rec)
    assert "mixed_sample_rates" in set(out["check"])
    assert out["severity"].iloc[0] == "bad"


def test_dead_channel_detected(faulty_rec):
    out = integrity.channel_alignment(faulty_rec)
    dead = out[out["check"] == "dead_channel"]
    assert set(dead["channel"]) == {"FLAT"}


def test_check_integrity_returns_empty_frame_when_clean(clean_rec):
    out = sq.check_integrity(
        clean_rec, checks=(integrity.coverage_gaps, integrity.channel_alignment)
    )
    assert len(out) == 0
    assert list(out.columns) == integrity.COLUMNS


def _with_stamps(rec, etc, first_stamp=0):
    return Recording(
        rec.ds,
        rec.source_path,
        rec.annotations,
        dict(rec.provenance, stamps=dict(etc=etc, stc=None), first_stamp=first_stamp),
        rec.defects,
    )
