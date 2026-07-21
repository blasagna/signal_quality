# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Dependencies are managed by [pixi](https://pixi.sh) (conda-forge, `linux-64`), not pip/venv. The
package itself is installed editable via `[pypi-dependencies]`.

```bash
pixi install                 # materialize env from pixi.lock
pixi run test                # pytest
pixi run lab                 # jupyter lab
pixi run pytest tests/test_filters.py::test_verdict_aggregates_to_worst_severity  # single test
pixi add <pkg>               # add a dep (updates pixi.toml + pixi.lock)
```

`pixi.lock` is marked `merge=binary linguist-generated=true` in `.gitattributes`: never hand-edit or
hand-merge it; regenerate via pixi and commit the result.

## Architecture

Five stages, deliberately decoupled — this separation is the point of the library, not incidental:

```
load(path) -> Recording -> compute(metrics, rec, grid) -> MetricFrame
           -> apply_filters -> flags -> verdict -> bad_segments -> viz
```

**The unit of judgement is the `(channel, interval)` cell**, at a default of 1-second
non-overlapping windows (`IntervalGrid.DEFAULT_WINDOW`). The deliverable is a set of *segments* to
exclude — "drop C3 from 412 s to 438 s" — not a list of condemned channels. `sq.assess(rec)` runs
the whole pipeline.

- **`core/recording.py`** — `Recording` wraps an `xarray.Dataset` (`signal`/`counts`/`covered` data
  vars; `ch_type`/`ch_unit`/`factor_uV` channel-coords; `sfreq`/`line_freq`/`meas_date` attrs).
  `build_dataset()` is the single construction path every reader goes through.
- **`core/intervals.py`** — `IntervalGrid` is built **once** and shared by all metrics, which is why
  joining them is a plain concat with no alignment guesswork. `.whole()`, `.fixed()`,
  `.from_annotations()`.
- **`core/context.py`** — `MetricContext` memoizes filtered signals and Welch spectra for the
  current *view* (whole recording, or one padded block). Metrics address samples in absolute
  recording coordinates; the context maps them into the view and raises if one reaches outside it.
- **`core/metric.py`** — `Metric.compute_interval()` returns one value per channel; `compute()`
  joins metrics into a `MetricFrame`. Metrics declare `requires` (e.g. `("counts",)`) and yield NaN
  plus a recorded note when a source can't satisfy them, rather than raising.
- **`core/blocks.py`** — block planning. `compute()` is block-major: each block loads a padded span,
  filters it once, computes every metric for its intervals, then releases the buffers.
- **`filters.py`** — thresholds live here, *not* in metrics, so the same computed table can be
  re-judged under a different policy for free. `verdict()` judges each cell; `channel_summary()`
  rolls up to `pct_bad`; `bad_segments()` merges runs into time spans.
- **`report.py`** — `assess()` runs both scales and combines them.

### MNE is an edge adapter only

The data model is xarray. MNE appears in a fixed set of edge modules, and
`tests/test_decoupling.py` enforces that (`ALLOWED` there lists each one with its justification):

- `io/load.py`, `io/xltek.py` — reading formats
- `core/context.py` — `mne.filter.filter_data` takes/returns plain ndarrays
- `core/blocks.py` — `mne.filter.create_filter`, to size block padding from the real FIR length
- `core/recording.py` — `to_mne()`, for notebook use only
- `export.py` — converts segments to MNE `Annotations` (kept out of `filters.py` for this reason)
- `montage.py`, `viz/topomap.py` — electrode geometry; `plot_topomap` takes a plain `(n,2)` array

**`metrics/` and `filters.py` must never import MNE.** Adding such an import fails the test suite.

### Two scales, and why both exist

Faults come at two scales and **neither subsumes the other**:

- **Channel-scope** (whole-recording pass, `WHOLE_RECORDING_FILTERS`) — sustained attenuation,
  isolation. Nearly invisible per-second: on the reference study an attenuated electrode scores
  robust z −8.9 over the recording but a median of only −1.2 per second, because the spread across
  channels *within one second* is far wider than the spread of their whole-recording averages.
- **Time-scope** (1-second pass, `DEFAULT_FILTERS`) — movement, a lead coming loose, transient
  saturation. A whole-recording summary averages these into a channel that merely looks mediocre.

`assess()` runs both and labels each finding with `scope`. `_drop_episode_artifacts()` stops
double-counting: a channel-scope finding is discarded when the channel's episodes are both
*concentrated* (span < 50% of the recording) and *substantial* (≥1% of it). Do **not** add a
"reasons must match" condition — the same event trips different metrics at different scales (a
movement episode reads `ARTIFACT` per second but `ISOLATED` over the recording), and requiring
matching labels keeps exactly the findings the rule exists to remove.

**Thresholds do not transfer between scales.** `DEFAULT_FILTERS` is calibrated for 1-second windows;
the whole-recording values would flag 87% of all cells. Both sets are validated against the
synthetic ground truth *and* the reference study, where the channel-scope pass reproduces the known
13-channel exclusion list exactly.

### Spectral metrics need a wider window than their interval

Frequency resolution is set by the analysed window. On a bare 1-second window the mains peak cannot
be separated from its own leakage: `line_ratio` for a bad electrode reads **5.1** instead of ~963,
with no error. So spectral metrics carry `min_analysis_s` (default 4.0) and compute their PSD over a
wider *centered* window while still attributing the value to the fine interval.

Consequence to state in any report: **onset timing from spectral flags is approximate** (to about
`min_analysis_s`); amplitude flags are exact to the interval.

### Chunking must never change a number

`Metric.required_pad_s()` **adds** the filter half-length and `min_analysis_s / 2` — they are not
alternatives. The widened analysis window reaches out from the interval, and every filtered sample
out there needs its own filter context beyond *that*. Taking the maximum instead made `emg_pct`
differ from whole-recording values on 29% of intervals. `tests/test_blocks.py` pins equivalence to
~1e-9 across block sizes; if it fails, chunking has silently changed every result.

### Metric vs filter — where does a check belong?

A **metric** measures one channel (or the channel set) from the signal. A **filter** compares
values in the finished table. Robust-z of RMS is a *filter* (`RobustZ`) because "outlier" is a
statement about a channel relative to its peers; `max_corr` is a *metric* because it needs the raw
data.

### Plotting conventions (`viz/`)

Every plot must state its y scale numerically. Two helpers in `viz/_scale.py` enforce this and
should be used by any new plot:

- `label_with_range(ax, values, label)` — appends the observed data range to the y label.
- `add_scale_bar(ax, size, unit)` — for stacked axes whose y ticks are channel names and therefore
  carry no amplitude information.

Other rules learned the hard way, each with a regression test:

- **`no_data` is not `good`.** Intervals inside a gap have nothing to judge; the heatmap gives them
  their own colour and `channel_summary` computes percentages over covered time only.
- **Never plot a value for an uncovered interval.** `_mask_uncovered()` blanks intervals inside a
  gap; without it `plot_clean_fraction` drew 0% clean, which reads as total artifact — the opposite
  of "nothing was recorded".
- **Robust colour limits on topomaps.** A dead channel reads 0, which is −inf once logged, and
  naive limits flatten every other electrode to one colour.
- **`plot_overview` renders min/max envelopes**, not decimated samples, or brief excursions vanish.
- **Normalize per channel when types are mixed** (`normalize="auto"`). EEG in µV and DC/position
  channels in device units cannot share an absolute scale — on the reference study the DC channels
  set the spacing and every EEG trace became a hairline. Note the trade-off: normalising by a
  channel's own SD *hides* a saturating channel, since its SD is inflated by the saturation.
- **Report persistently off-scale channels by clipped fraction, not peak.** Keying on peak named 37
  of 50 channels on real data, because electrode pops are ubiquitous.
- **Rank flagged examples by severity**, not alphabetically, so evidence plots show the worst cases.
- **`plot_verdict_topomap` refuses per-interval input.** A head plot shows one value per electrode;
  which value that should be is policy, so it must go through `channel_summary` rather than be
  guessed in the plotting layer.

### Degenerate channels

A dead channel has *no* correlation and *no* spectral ratio — both normalise by a quantity that is
numerical residue, producing finite but arbitrary values that then propagate into other channels'
statistics and make them depend on floating-point noise. `MaxCorrelation.dead_rel_tol` and
`spectral._dead()` detect this relatively (so it holds in any units) and return NaN. This was found
because it made `max_corr` differ across block sizes.

### Recording-scope checks are separate

`metrics/integrity.py` (gaps, timestamp anomalies, channel alignment) emits its own findings table
(`check/severity/t_start/t_end/channel/detail`) rather than joining the per-channel frame — a gap is
a property of the recording, and broadcasting it across every channel row would be noise.

## Domain notes worth knowing

- **`line_freq` is a parameter, not 60.** Defaults to the recording's `line_freq` attr; 50 Hz
  outside the Americas.
- **Legacy 10-20 labels.** T3/T4/T5/T6 are still used clinically but absent from standard montages.
  `montage.place()` handles the rename and returns positions under the *original* labels so they
  still match the metric table. Do not re-implement this inline.
- **Gaps must be masked, not analysed.** They are zero-filled; every metric masks to `covered`.
- **No automatic bridge detection.** Near-unity correlation looks like a salt bridge, but on a
  common-reference recording the shared reference and drift push nearly every pair above 0.97 —
  measured median `max_corr` on a real study was 0.984, and electrical distance (`var(a-b)`) fails
  the same way. `DEFAULT_FILTERS` therefore ships no `BRIDGED` filter; `correlation_pairs()`
  shortlists candidates for a human. Don't "fix" this by adding the threshold back.
- **Raw ADC counts matter.** Clipping is only detectable in integer counts — once scaled to volts
  the rail is no longer a distinguished value. Only XLTEK studies and the lossless HDF5 keep them.

## Testing

Two layers of synthetic data, deliberately separate:

- **`tests/conftest.py`** — minimal fixtures, one fault each, for unit tests.
- **`signal_quality/synthetic.py`** — `make_demo_recording()`, a full 10-20 montage carrying every
  fault at once plus a ground-truth table. Public API, and the notebook's default input.
  `tests/test_synthetic.py` scores detection against that truth and fails if the generator drifts.

Signal-model constants there are load-bearing and were tuned empirically; changing them casually
will break detection in non-obvious ways:

- **Spectral slope** (`_pink` exponent) — shallower noise puts so much power above 25 Hz that every
  channel looks full of muscle artifact.
- **Common/independent ratio** (0.95 / 0.30) — referential EEG correlates at ~0.9+; too little
  shared signal and healthy channels trip `ISOLATED`.
- **Per-site gain jitter** — without it every healthy channel has near-identical amplitude, the MAD
  collapses, and `RobustZ` flags ordinary channels over trivial differences.
- **EMG must be band-limited** (`_band_noise`, 25–45 Hz) — broadband noise spreads power outside the
  measured band, so raising `emg_pct` enough would decorrelate the channel into `ISOLATED` instead.
- **Clipping is modelled as slow excursions**, not impulses — one-sample spikes at rail magnitude
  dominate a channel's variance so completely it also reads as uncorrelated.

Some flags are legitimate *side effects*, not false positives: a dead or rail-swinging channel
genuinely correlates with nothing, so `T5`/`Fpz` also trip `ISOLATED` (see `EXPECTED_SIDE_EFFECTS`).

Real recordings are patient data and stay **outside** this repo. Do not add data files, subject
paths, or identifiers. Notebook outputs must be cleared or de-identified before sharing.

The library was validated against a reference XLTEK study: it independently reproduced that study's
hand-derived 13-channel exclusion list exactly, along with its line-noise set, C3 attenuation
(robust z ≈ −9), 5 clipping channels, A1 isolation (max_corr 0.53), and 6.6-minute gap.
