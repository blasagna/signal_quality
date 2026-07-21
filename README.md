# signal_quality

Signal quality checks and visualizations in python, focused on use from jupyter notebooks. Data will
be kept outside of this repo and presented deidentified in any artifacts.

Each quality check is an independent **metric** computed over a shared grid of **intervals**.
Metrics join into one table. **Filters** are defined separately and applied to that table, so the
same measurements can be re-judged under a different policy without recomputing anything.

```python
import signal_quality as sq
from signal_quality import metrics as M

rec  = sq.load("/path/to/study")            # XLTEK dir, lossless .h5, or anything MNE reads
grid = sq.IntervalGrid.whole(rec)           # or .fixed(rec, 30.0) for a trend

mf = sq.compute([M.RMS(), M.LineRatio(), M.EMGFraction(),
                 M.MaxCorrelation(), M.FlatFraction(), M.ClipFraction()], rec, grid)

issues   = sq.check_integrity(rec)                    # gaps, clock, channel alignment
flags    = sq.apply_filters(mf, sq.DEFAULT_FILTERS)   # policy, applied after the fact
verdicts = sq.verdict(flags)                          # good / marginal / bad per channel

sq.viz.plot_overview(rec)                             # all raw signals, whole recording
sq.viz.plot_contact_quality(mf, verdicts)             # electrodes on the head model
sq.viz.plot_good_bad_psd(rec, flags, "LINE_NOISE")    # spectra: flagged vs clean
```

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

Injected: mains pickup (F9/F10/T9/T10/P9/P10), an attenuated contact (C3), a dead electrode (T5) and
a never-recorded aux channel (OSAT), an isolated one (A1), converter clipping plus oversized
amplitude (Fpz), muscle contamination (F7/F8), a mid-recording gap, and a corrupted acquisition
clock. The notebook scores the run at **12/12 injected faults detected, 0 false positives**.

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
- The data model is `xarray`; MNE is used only for reading files, array-level filtering, and
  electrode geometry, so the metrics are not EEG-specific.
