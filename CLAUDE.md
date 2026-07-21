# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Greenfield. As of this writing the repo contains only packaging (`pixi.toml`), `README.md`, and license/ignore files — no Python source, tests, or notebooks yet. Treat structural conventions below as decisions still to be made, not as existing facts to conform to.

## Environment

Dependencies are managed by [pixi](https://pixi.sh) (conda-forge only, `linux-64` only), not pip/venv. The environment lives in `.pixi/envs/default/`.

```bash
pixi install                 # materialize env from pixi.lock
pixi run python -c "..."     # run anything inside the env
pixi run jupyter lab         # primary interactive entry point
pixi add <pkg>               # add a dep (updates pixi.toml + pixi.lock)
pixi shell                   # interactive shell in the env
```

`[tasks]` in `pixi.toml` is empty. When adding build/lint/test entry points, define them there (`pixi run test`, etc.) rather than expecting bare `pytest`/`ruff` on PATH. No test runner is currently a dependency — adding one means `pixi add pytest` first.

`pixi.lock` is marked `merge=binary linguist-generated=true` in `.gitattributes`: never hand-edit or hand-merge it; regenerate via pixi and commit the result.

## Domain context

Signal quality checks and visualizations for physiological time series, designed to be driven from Jupyter notebooks. MNE-Python is a core dependency, so EEG/MEG-style data (`Raw`, channel/annotation metadata, sfreq) is the expected input shape.

Planned check families, per README:
- Generic: data existence/gaps, valid time points
- Signal-specific: flatline (no time-varying signal), railing at min/max, power line interference (60 Hz in America — make line frequency a parameter, not a constant), intermittent contact quality / motion artifacts

Because the checks are notebook-facing, prefer a library of small pure functions that take arrays or MNE objects and return structured results, with plotting kept separate from computation so a check can be used headlessly or rendered.

## Data handling

Recordings are kept **outside** this repo and are patient data. Do not add data files, paths to specific subjects, or subject identifiers to the repo. Anything produced for sharing — figures, notebook outputs, example fixtures — must be deidentified.
