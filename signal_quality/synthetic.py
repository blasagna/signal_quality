"""Synthetic recordings with known, deliberately injected faults.

Real recordings are patient data and live outside this repo, which makes the
library awkward to demonstrate and impossible to show in a shareable artifact.
This module builds a recording that carries every fault the library checks for,
together with a ground-truth table saying which channel got which — so a
demonstration can be *scored* rather than merely admired.

It is a teaching and testing aid, not a simulator: the signal is coloured noise
shaped to have roughly the spectral character of scalp EEG, not a model of
cortical activity.

    rec, truth = make_demo_recording()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core.recording import Recording, build_dataset

#: Device constant borrowed from a real amplifier, so counts↔µV scaling and the
#: converter rail sit at realistic magnitudes.
FACTOR_UV = 0.2658386864277569
RAIL = 131071  # 2**17 - 1

#: Standard 10-20 sites, the extended/inferior chain, ears, and two aux
#: channels. The extended electrodes are the ones that go wrong in practice —
#: they are harder to apply and are exactly where mains pickup shows up.
CORE_1020 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
             "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
EXTENDED = ["F9", "F10", "T9", "T10", "P9", "P10", "Fpz"]
EARS = ["A1", "A2"]
AUX = ["ECGL", "OSAT"]


def _pink(rng, n_ch, n, sfreq, exponent=1.0):
    """Noise with a 1/f^exponent amplitude spectrum.

    The slope matters: with shallower noise, high-frequency power dominates and
    every channel would look like it is full of muscle artifact.
    """
    spec = np.fft.rfft(rng.standard_normal((n_ch, n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / sfreq)
    return np.fft.irfft(spec / np.maximum(f, 0.5) ** exponent, n=n, axis=1)


def _band_noise(rng, n, sfreq, lo, hi):
    """Noise confined to ``[lo, hi]`` Hz.

    Broadband noise would spread most of its power outside the band a metric
    looks at, so raising that metric would require so much of it that the
    channel decorrelates from the montage and reads as isolated instead.
    """
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sfreq)
    spec[(f < lo) | (f > hi)] = 0
    out = np.fft.irfft(spec, n=n)
    return out / (out.std() + 1e-12)


def _alpha(rng, n_ch, n, sfreq, freq=10.0):
    """A wandering ~10 Hz rhythm, so the spectra have a recognisable peak."""
    t = np.arange(n) / sfreq
    phase = np.cumsum(rng.standard_normal((n_ch, n)) * 0.05, axis=1)
    envelope = 0.5 + 0.5 * np.abs(_pink(rng, n_ch, n, sfreq, exponent=2.0))
    envelope /= envelope.std(axis=1, keepdims=True) + 1e-12
    return np.sin(2 * np.pi * freq * t + phase) * np.abs(envelope)


def make_demo_recording(seed: int = 0, duration: float = 180.0,
                        sfreq: float = 250.0, line_freq: float = 60.0,
                        with_gap: bool = True, with_stamp_fault: bool = True):
    """Build a demo :class:`~signal_quality.core.recording.Recording`.

    Returns ``(rec, truth)``. ``truth`` is a **time-resolved** ground truth —
    ``(channel, t_start, t_end, injected)`` — so detection can be scored on
    *when* as well as *which*, which is what per-interval flagging claims to do.

    Sustained faults, spanning the whole recording:

    ==================  ==========================================
    ``LINE_NOISE``      F9/F10/T9/T10/P9/P10 — mains pickup
    ``AMP_OUTLIER``     C3 attenuated; Fpz far too large
    ``FLAT``            T5 dead; OSAT never recorded
    ``ISOLATED``        A1 shares no signal with the montage
    ``CLIPPING``        Fpz saturates the converter
    ``EMG``             F7/F8 muscle contamination
    ==================  ==========================================

    Episodic faults, bounded in time — these are the ones a per-channel verdict
    cannot express, since the channel is perfectly usable outside its episode:

    ==================  ==========================================
    ``LINE_NOISE``      Fp1 comes loose partway through
    ``ARTIFACT``        O1/O2/Pz movement episode, several channels at once
    ``CLIPPING``        T4 brief saturating burst
    ``FLAT``            P3 disconnects, then is reseated
    ==================  ==========================================

    Plus recording-scope faults: a mid-recording gap, and a nonmonotonic
    acquisition stamp that is *not* explained by that gap.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sfreq)
    names = CORE_1020 + EXTENDED + EARS + AUX
    n_ch = len(names)
    t = np.arange(n) / sfreq

    # --- plausible background: a strongly shared component plus local activity
    # Referential scalp EEG is dominated by the shared reference and by common
    # drift, so healthy channels correlate at ~0.9+. Getting this ratio wrong
    # makes ordinary channels look isolated.
    common = _pink(rng, 1, n, sfreq)
    common /= common.std()
    X = _pink(rng, n_ch, n, sfreq)
    X /= X.std(axis=1, keepdims=True)
    X = 0.30 * X + 0.95 * common
    X *= 20.0 / X.std(axis=1, keepdims=True)          # ~20 µV RMS

    # Per-site gain variation. Without it every healthy channel has almost
    # exactly the same amplitude, the median absolute deviation collapses, and
    # the robust z-score flags ordinary channels as outliers over differences
    # far too small to matter.
    X *= np.exp(rng.normal(0.0, 0.22, size=(n_ch, 1)))

    # One posterior alpha *source*, projected onto posterior sites with
    # different gains — a shared rhythm, not independent noise per electrode.
    posterior = {"O1": 1.0, "O2": 0.95, "P3": 0.7, "P4": 0.65, "Pz": 0.5}
    alpha = _alpha(rng, 1, n, sfreq)[0]
    alpha /= alpha.std() + 1e-12
    for ch, gain in posterior.items():
        if ch in names:
            X[names.index(ch)] += 9.0 * gain * alpha

    # Ground truth is time-resolved: (channel, t_start, t_end, fault). A fault
    # that lasts the whole recording spans [0, duration]; an episodic one spans
    # only its episode, which is what makes per-interval detection scoreable.
    truth = []

    def mark(ch, fault, t0=0.0, t1=None):
        truth.append(dict(channel=ch, t_start=float(t0),
                          t_end=float(duration if t1 is None else t1),
                          injected=fault))

    def span(t0, t1):
        return slice(int(t0 * sfreq), int(t1 * sfreq))

    # --- injected per-channel faults ----------------------------------------
    # Amplitudes chosen so the resulting line ratios land in the same range as
    # real poorly-applied electrodes on the reference study (~11,000 against
    # <900 for clean ones), so the demo exercises the shipped thresholds rather
    # than a softer set of its own.
    for ch, amp in (("F9", 78.0), ("F10", 72.0), ("T9", 90.0),
                    ("T10", 66.0), ("P9", 75.0), ("P10", 60.0)):
        i = names.index(ch)
        X[i] += amp * np.sin(2 * np.pi * line_freq * t + rng.uniform(0, 6.28))
        mark(ch, "LINE_NOISE")

    i = names.index("C3")                              # attenuated contact
    X[i] *= 0.18
    mark("C3", "AMP_OUTLIER")

    i = names.index("T5")                              # dead electrode
    X[i] = 0.0
    mark("T5", "FLAT")

    i = names.index("OSAT")                            # never recorded
    X[i] = 0.0
    mark("OSAT", "FLAT")

    i = names.index("A1")                              # floating / isolated
    A1 = _pink(rng, 1, n, sfreq)[0]
    # Matched to the montage's typical amplitude, so this channel is isolated
    # and *only* isolated — keeping the injected faults separable.
    X[i] = A1 / A1.std() * float(np.median(np.abs(X).std(axis=1)))
    mark("A1", "ISOLATED")

    for ch in ("F7", "F8"):                            # muscle contamination
        i = names.index(ch)
        # Scaled to this channel's own amplitude: EMG *fraction* is what the
        # metric measures, so a fixed µV level would land differently on every
        # channel once per-site gain variation is applied.
        X[i] += 0.28 * X[i].std() * _band_noise(rng, n, sfreq, 25.0, 45.0)
        mark(ch, "EMG")

    # Oversized channel that intermittently saturates the converter. Modelled as
    # brief slow movement excursions whose peaks graze the rail, not as impulse
    # spikes: a real electrode clips because the signal swings out of range, and
    # isolated one-sample impulses would dominate the channel's variance so
    # completely that it would also read as uncorrelated with the whole montage.
    i = names.index("Fpz")
    X[i] *= 3.0
    rail_uV = RAIL * FACTOR_UV
    for s in rng.integers(0, n - int(2.5 * sfreq), size=3):
        w = int(rng.integers(int(0.15 * sfreq), int(0.4 * sfreq)))
        # Peak only just past the rail, so the excursion's tip clips rather
        # than most of its length.
        bump = np.hanning(w) * rng.choice([-1.0, 1.0]) * rail_uV * 1.08
        X[i, s:s + w] += bump
    mark("Fpz", "AMP_OUTLIER")
    mark("Fpz", "CLIPPING")

    # --- episodic faults, the reason for flagging per interval --------------
    # Each is bounded in time, so a per-channel verdict could only say "this
    # electrode is suspect" while the truth is "this electrode was fine until
    # 90 s". Episodes are placed clear of the recording gap so they are
    # observable.
    ep = _episodes(duration)

    # An electrode that works, then comes loose and starts picking up mains.
    # Amplitude matched to the sustained line-noise channels above: a genuinely
    # floating electrode is an antenna, not a marginal case.
    t0, t1 = ep["pop"]
    i = names.index("Fp1")
    X[i, span(t0, t1)] += 78.0 * np.sin(
        2 * np.pi * line_freq * t[span(t0, t1)] + rng.uniform(0, 6.28))
    mark("Fp1", "LINE_NOISE", t0, t1)

    # A movement/muscle episode across several channels at once — the signature
    # of the subject, not of any one electrode.
    t0, t1 = ep["movement"]
    for ch in ("O1", "O2", "Pz"):
        i = names.index(ch)
        seg = span(t0, t1)
        X[i, seg] += 1.1 * X[i].std() * _band_noise(
            rng, n, sfreq, 25.0, 45.0)[seg]
        X[i, seg] += 220.0 * _pink(rng, 1, n, sfreq, exponent=2.0)[0][seg] / (
            _pink(rng, 1, n, sfreq, exponent=2.0)[0].std() + 1e-12)
        mark(ch, "ARTIFACT", t0, t1)

    # A brief saturating burst: fine before and after.
    t0, t1 = ep["burst"]
    i = names.index("T4")
    sl = span(t0, t1)                       # length from the slice, not recomputed:
    w = sl.stop - sl.start                  # the two round differently
    X[i, sl] += np.hanning(w) * (RAIL * FACTOR_UV) * 1.1
    mark("T4", "CLIPPING", t0, t1)

    # An electrode that disconnects and is reseated.
    t0, t1 = ep["dropout"]
    i = names.index("P3")
    X[i, span(t0, t1)] = 0.0
    mark("P3", "FLAT", t0, t1)

    # --- quantise to ADC counts, which is where clipping actually happens ----
    factor_uV = np.full(n_ch, FACTOR_UV)
    counts = np.clip(np.rint(X / factor_uV[:, None]), -RAIL, RAIL).astype(np.int32)
    signal = counts.astype(np.float64) * factor_uV[:, None] * 1e-6   # -> volts

    # --- recording-scope faults ---------------------------------------------
    covered = np.ones(n, dtype=bool)
    annotations = []
    if with_gap:
        g0, g1 = int(0.42 * n), int(0.55 * n)
        covered[g0:g1] = False
        counts[:, g0:g1] = 0
        signal[:, g0:g1] = 0.0
        annotations.append((g0 / sfreq, (g1 - g0) / sfreq, "BAD_gap"))

    ch_types = ["eeg"] * (len(CORE_1020) + len(EXTENDED) + len(EARS))
    ch_types += ["ecg", "misc"]

    ds = build_dataset(signal, sfreq, names, ch_types,
                       ch_units=["uV"] * n_ch, counts=counts,
                       factor_uV=factor_uV, covered=covered,
                       meas_date=None, line_freq=line_freq)

    provenance = {"reader": "synthetic", "seed": seed, "first_stamp": 0}
    if with_stamp_fault:
        provenance["stamps"] = {"etc": _stamp_table(n, sfreq, covered), "stc": None}

    ann = pd.DataFrame(annotations, columns=["onset", "duration", "description"])
    rec = Recording(ds, None, ann, provenance, defects=[])

    truth_df = pd.DataFrame(
        truth, columns=["channel", "t_start", "t_end", "injected"]
    ).sort_values(["channel", "t_start"], ignore_index=True)
    return rec, truth_df


def _episodes(duration: float) -> dict:
    """Where the time-localized faults live, as fractions of the recording.

    Kept clear of the gap at 0.42–0.55 so each episode is actually observable;
    a fault injected into missing data would be untestable.
    """
    d = duration
    return {
        "movement": (0.20 * d, 0.28 * d),
        "burst": (0.33 * d, 0.35 * d),
        "pop": (0.62 * d, 0.78 * d),
        "dropout": (0.85 * d, 0.95 * d),
    }


def _stamp_table(n, sfreq, covered):
    """Acquisition stamp table with one genuine clock fault.

    Packets skip the recording gap — which must *not* be reported as a clock
    anomaly, since it is already reported as missing data — and one packet
    carries a stamp that jumps backwards, which must be.
    """
    dt = np.dtype([("offset", "<i4"), ("samplestamp", "<i4"),
                   ("sample_num", "<i4"), ("sample_span", "<i2"),
                   ("unknown", "<i2")])
    span = 250
    edges = np.diff(covered.astype(np.int8))
    starts = np.where(edges == -1)[0] + 1
    ends = np.where(edges == 1)[0] + 1
    g0 = int(starts[0]) if len(starts) else n
    g1 = int(ends[0]) if len(ends) else n

    stamps = np.concatenate([np.arange(0, g0 - span, span),
                             np.arange(g1, n - span, span)])
    etc = np.zeros(len(stamps), dtype=dt)
    etc["samplestamp"] = stamps
    etc["sample_span"] = span
    if len(etc) > 6:
        etc["samplestamp"][5] = max(0, int(etc["samplestamp"][3]) - span)
    return etc
