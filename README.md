# signal_quality

Signal quality checks and visualizations in python, focused on use from jupyter notebooks. Data will
be kept outside of this repo and presented deidentified in any artifacts.

Quality is judged per **`(channel, interval)`** cell — by default 1-second non-overlapping windows —
so the answer is *which channels are bad and when*. The deliverable is a set of time spans to
exclude, not a list of condemned electrodes.

```python
import signal_quality as sq

rec = sq.load("/path/to/study")     # XLTEK dir, lossless .h5, or anything MNE reads
report = sq.assess(rec)             # 1-second windows by default

report.channels          # per channel: pct_bad, pct_marginal, pct_no_data, verdict
report.segments          # what to exclude: (channel, t_start, t_end, reasons, scope)
report.bad_channels      # channels with a *sustained* defect
sq.to_annotations(report.segments)  # channel-scoped MNE annotations

sq.viz.plot_quality_heatmap(report.verdicts, report.metrics)   # channels x time
sq.viz.plot_pct_bad_topomap(report.channels)                   # on the head model
```

Each stage is available separately if you want to vary the metrics, grid, or thresholds:

```python
from signal_quality import metrics as M

grid  = sq.IntervalGrid.fixed(rec, 1.0)
mf    = sq.compute([M.RMS(), M.LineRatio(), M.PeakToPeak()], rec, grid)
flags = sq.apply_filters(mf, sq.DEFAULT_FILTERS)     # policy, applied after the fact
v     = sq.verdict(flags, mf)                        # per (channel, interval)
segs  = sq.bad_segments(v, mf, min_duration=2.0)     # merged into time spans
```

## Two scales

Faults come at two scales and neither subsumes the other, so `assess()` runs both and tags each
finding with its `scope`:

* **Channel-scope** — sustained attenuation or isolation. Nearly invisible per second: an electrode
  attenuated to a third of its neighbours scores a robust z of −8.9 over a recording but only −1.2
  per second, because channels differ far more within one second than their averages do.
* **Time-scope** — movement, a lead coming loose, a transient saturation. A whole-recording summary
  averages these away into a channel that merely looks mediocre.

Plots state their y scale numerically — an explicit data range on ordinary axes, and a µV scale bar
on stacked trace plots, whose y ticks carry channel names rather than values.

See `notebooks/quality_report.ipynb` for the full report: load → metrics → filters → a visualization
for every flagged issue. It runs out of the box on a synthetic recording, so no data is needed to
try it.

## Try it without any data

`make_demo_recording()` builds a recording carrying a known fault of every kind the library checks
for, and returns the ground truth alongside it — so detection can be *scored* rather than eyeballed:

```python
rec, truth = sq.make_demo_recording()
```

Ground truth is **time-resolved** — `(channel, t_start, t_end, injected)` — so detection is scored on
*when* as well as *which*.

*Sustained*: mains pickup (F9/F10/T9/T10/P9/P10), an attenuated contact (C3), a dead electrode (T5)
and a never-recorded aux channel (OSAT), an isolated one (A1), converter clipping plus oversized
amplitude (Fpz), muscle contamination (F7/F8).

*Episodic* — the faults a per-channel verdict cannot express, since the channel is fine outside its
episode: Fp1 comes loose partway through, a movement episode hits O1/O2/Pz at once, T4 saturates
briefly, P3 disconnects and is reseated. Plus a mid-recording gap and a corrupted acquisition clock.

The notebook scores the run at **19/19 injected faults detected, 0 clean channels condemned**.

It is a teaching and testing aid, not a simulator: the signal is coloured noise shaped to resemble
scalp EEG, not a model of cortical activity.

## Checks

**Generic** (recording-scope, signal-agnostic, in their own findings table)
- data existence and gaps, with total missing fraction
- timestamp anomalies — nonmonotonic stamps, overlapping packets, irregular sample periods
- channel alignment — mixed per-channel sample rates, shorted channels, dead channels

**Per-channel**
- no time-varying signal (`flat_frac`)
- railing at the converter's minimum or maximum (`clip_pct`, needs raw ADC counts)
- power line interference (`line_ratio`; 60 Hz in the Americas, 50 elsewhere — a parameter)
- intermittent contact quality and motion artifacts (`rms`, `emg_pct`, `max_corr`, `p2p`)

Amplitude outliers are found with a robust (median/MAD) z-score across channels, which is a
*filter* rather than a metric because being an outlier is a statement about a channel relative to
its peers.

## Install

```bash
pixi install
pixi run test
pixi run lab
```

## Notes and limitations

- **Bridging is not auto-detected.** Near-unity correlation is consistent with a salt bridge, but on
  a common-reference recording the shared reference and drift push nearly every pair above 0.97, so
  a threshold flags most of the montage while still missing real bridges.
  `metrics.correlation_pairs()` shortlists candidates; confirm them against a bipolar derivation.
- **Clipping needs raw integer counts.** Once data is scaled to volts the converter rail is no
  longer identifiable, so this metric reports itself unavailable rather than guessing.
- **Spectral flag onsets are approximate.** `line_ratio` and `emg_pct` need ≥2 s to resolve a mains
  peak from its own leakage, so they compute over a wider centered window (`min_analysis_s`, default
  4 s) and their onsets are accurate only to about that. Amplitude flags are exact to the interval.
- **Thresholds are window-length specific.** `DEFAULT_FILTERS` is calibrated for 1-second windows;
  `WHOLE_RECORDING_FILTERS` for `IntervalGrid.whole`. They are not interchangeable — the
  whole-recording values applied per second would flag 87% of all cells.
- **Memory is bounded for derived arrays, not the source.** Block processing keeps filtered copies
  and spectra small, but the whole recording is still held in memory, so a multi-hour study will not
  fit. Lazy loading is the follow-on.
- The data model is `xarray`; MNE is used only for reading files, array-level filtering, and
  electrode geometry, so the metrics are not EEG-specific.

## TODO

- video quality: continuity, frame rate, dynamic range
- EKG
- EMG
- IMU
