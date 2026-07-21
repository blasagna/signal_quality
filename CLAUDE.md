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

Four stages, deliberately decoupled — this separation is the point of the library, not incidental:

```
load(path) -> Recording -> compute(metrics, rec, grid) -> MetricFrame -> apply_filters -> flags -> viz
```

- **`core/recording.py`** — `Recording` wraps an `xarray.Dataset` (`signal`/`counts`/`covered` data
  vars; `ch_type`/`ch_unit`/`factor_uV` channel-coords; `sfreq`/`line_freq`/`meas_date` attrs).
  `build_dataset()` is the single construction path every reader goes through.
- **`core/intervals.py`** — `IntervalGrid` is built **once** and shared by all metrics, which is why
  joining them is a plain concat with no alignment guesswork. `.whole()`, `.fixed()`,
  `.from_annotations()`.
- **`core/context.py`** — `MetricContext` memoizes filtered signals and Welch spectra per
  `(interval, params)`. Without it, modular metrics would refilter the whole array once each.
- **`core/metric.py`** — `Metric.compute_interval()` returns one value per channel; `compute()`
  joins metrics into a `MetricFrame`. Metrics declare `requires` (e.g. `("counts",)`) and yield NaN
  plus a recorded note when a source can't satisfy them, rather than raising.
- **`filters.py`** — thresholds live here, *not* in metrics, so the same computed table can be
  re-judged under a different policy for free.

### MNE is an edge adapter only

The data model is xarray. MNE appears in exactly five modules, and `tests/test_decoupling.py`
enforces that (`ALLOWED` there lists each one with its justification):

- `io/load.py`, `io/xltek.py` — reading formats
- `core/context.py` — `mne.filter.filter_data` takes/returns plain ndarrays
- `core/recording.py` — `to_mne()`, for notebook use only
- `montage.py`, `viz/topomap.py` — electrode geometry; `plot_topomap` takes a plain `(n,2)` array

**`metrics/` and `filters.py` must never import MNE.** Adding such an import fails the test suite.

### Metric vs filter — where does a check belong?

A **metric** measures one channel (or the channel set) from the signal. A **filter** compares
values in the finished table. Robust-z of RMS is a *filter* (`RobustZ`) because "outlier" is a
statement about a channel relative to its peers; `max_corr` is a *metric* because it needs the raw
data.

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

Synthetic fixtures in `tests/conftest.py` inject one fault each (flat, mains tone, ADC rail, bridge,
amplitude outlier, gap, corrupt stamp tables). Note `_pink()`'s spectral slope is load-bearing: with
shallower noise, high-frequency power dominates and every channel looks full of muscle artifact.

Real recordings are patient data and stay **outside** this repo. Do not add data files, subject
paths, or identifiers. Notebook outputs must be cleared or de-identified before sharing.

The library was validated against a reference XLTEK study: it independently reproduced that study's
hand-derived 13-channel exclusion list exactly, along with its line-noise set, C3 attenuation
(robust z ≈ −9), 5 clipping channels, A1 isolation (max_corr 0.53), and 6.6-minute gap.
