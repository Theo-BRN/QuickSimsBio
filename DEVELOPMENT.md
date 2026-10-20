# Development guide

These Maintainer-facing notes for `quicksimsbio` describe the design rationale, internal architecture, and underlying decisions. Start here if you're inheriting or extending the package. For _usage_ of the repo, see the [README](README.md).

## What this package is for

QuickSimsBio wraps around [basico](https://basico.readthedocs.io/) (the Python interface to COPASI) to make ODE / systems-biology simulation, parameter scanning, and quasi-random sampling convenient and simple to implement. Any model that basico can load can be used by `quicksimsbio`; any COPASI `.cps` files, sbml files, or files via biomodels ID.

QuickSimsBio is also the underlying package used for QuickSimsBio-UI, a streamlit-based graphic user interface (GUI) application. The UI depends on this package's public functions and output formats. If you are developing both, check changes to public signatures or outputs against the UI; if you are only working on this package, it need not constrain you.

Design philosophy: **simple, robust, working** over flashy; (working solidly) > (flashy and incomprehensible).

## Project structure

The package is two modules:

| File                                | Role                                                                                                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/quicksimsbio/sampling.py`      | **Core module.** Explicit-model API: simulation, `scan_dict` sampling, and the dependency-resolution pipeline behind `auto_melt`. All new simulation/sampling work goes here. |
| `src/quicksimsbio/curve_fitting.py` | **Stable and standalone.** scipy curve fitting; no basico dependency.                                                                                                         |

The public API is exposed via `__init__.py` to allow curated tools to appear in autocomplete: callers do `from quicksimsbio import run_simulations`, and internal plumbing (underscore-prefixed, e.g. `_generate_parameter_sets_df`) stays hidden.

Package source uses the `src/` (PEP 517) layout. No linter or formatter is configured.

## Public API (the stable tools)

These five functions are what users should call, and the names exported from `__init__.py`. Internal helpers including `auto_melt`, `get_dependency_status_dataframe`, `_generate_parameter_sets_df`, etc.) are **not** part of the user-facing API. If users wish to transform wide-format dataframes to long-format dataframes (melting), they should `pd.melt` manually - only reaching for `auto_melt` when the data structure is already well understood.

- **`get_model_inputs(model=None)`** — returns a scan dictionary '`scan_dict`' skeleton of all model inputs (all keys `None`) created via a
  temporary basico parameter set.
- **`generate_input_space(scan_dict, n_rand_samples)`** — quasi-random (Sobol/Halton/LHS) + grid sampling.
- **`run_simulations(scan_dict, timepoints, output_selection, ..., model=None)`** — runs all parameter combinations from a `scan_dict`; returns a wide (default) or long (`format_output="long"`, via `auto_melt`) DataFrame.
- **`get_basico_outputs(model=None)`** — inspect what output column names a model can use; useful when building `output_selection` lists.
- **`fit_curve_to_data(x, y, model=...)`** — simple scipy curve fitting. Supports `"3 parameter logistic"`, `"4 parameter logistic"`, `"rectangular_hyperbola"`, `"straight_line"`.

## Core design decisions

### Explicit model argument

All functions in `sampling.py` accept `model: bsc.COPASI.CDataModel | None = None`. When `None`, they fall back to `bsc.get_current_model()`. Pass `model=dm` explicitly wherever possible to avoid global-state issues.

### Simulation engine

`bsc.run_time_course_with_output()` is the only time-course function used intentionally. Other basico task types may well be eventually incorporated.

### Dependency classification (`auto_melt`)

`auto_melt` automatically melts a wide-format dataframe to a long format dataframe, using calculated independent and dependent variables. To calculate these variables, `auto_melt` builds a `status_df` that classifies every possible model output as `independent` or `dependent`. Pipeline: `get_dependency_status_dataframe()` → `create_status_df()` + `resolve_status_df()`. The logic handles:

- Fixed parameters → independent
- Transient species → dependent
- Local reaction parameters → independent
- Assigned parameters/compartments → resolved iteratively by inspecting their expressions for dependent constituents
- Mapped reaction parameters → resolved the same way, treating the global quantity they are mapped to as a one-term expression (e.g. `(R1).k1` → `Values[kg]`)

This lets `auto_melt` work without the user specifying `id_vars`. It is deliberately strict rather than approximate — if developing this, please avoid simplifying it away.

### Reaction parameters (rate constants)

basico reports reaction parameters via `bsc.get_reaction_parameters()`, separately from global quantities (`bsc.get_parameters()`). They are named `(ReactionName).ParamName`, e.g. `(R1).k1`, and that same name is used as the `scan_dict` key, for setting the value, and as the output column. Each one is either:

- **local** — a constant belonging to that one reaction (COPASI does not allow assignments or ODEs on local parameters). Always independent and directly settable.
- **mapped** (basico `type == "global"`, with `mapped_to` naming the global) — the reaction uses a global quantity instead of a local value, so it takes that global's value and inherits its fixed/assigned status.

Mapped reaction parameters are handled with care:

- `get_model_inputs` excludes them automatically: the temporary parameter set marks them `simulation_type: "assignment"`, and only `"fixed"` entries are kept.
- `set_model_parameters` skips them if they are supplied anyway. **Calling `bsc.set_reaction_parameters` on a mapped parameter silently changes the global quantity it is mapped to** (and so every reaction using it).
- `run_simulations` logs one warning per run (not per simulation) naming the global to vary instead, via `_warn_mapped_reaction_params`. This was originally a warning rather than an error to not interrupt callers such as QuickSimsBio-UI.

### Quasi-random sampling (`scan_dict` format)

`scan_dict` values can be:

- `None` — hold at model's current value (default or what it was last changed to)
- `list` — discrete values (grid scan)
- `dict` with `{"lower": ..., "upper": ..., "log": bool}` — quasi-random range

`generate_input_space(scan_dict, n_rand_samples)` returns a wide DataFrame of parameter combinations (grid × Quasi random sampling).

### Leaning on basico

This package is not to be a replacement for basico, it should instead work well with it. Hence where possible, use basico-native functions to extract model information. `get_model_inputs` is a good example — it uses a temporary `bsc.add_parameter_set()` call rather than reimplementing parameter discovery. Ideally, follow this kind of pattern when possible.

## Output column naming (basico conventions)

| Variable type                   | `output_selection` format        |
| ------------------------------- | -------------------------------- |
| Transient species concentration | `[SpeciesName]`                  |
| Initial species concentration   | `[SpeciesName]_0`                |
| Fixed parameter (transient)     | `Values[ParamName]`              |
| Fixed parameter (initial)       | `Values[ParamName].InitialValue` |
| Reaction (local) parameter      | `(ReactionName).ParamName`       |
| Reaction flux                   | `(ReactionName).Flux`            |
| Time                            | `Time`                           |

(Though follow basico documentation if there is a conflict).

## Tests

Regression tests live in `tests/test_api.py` and cover the public API across four bundled basico example models (`brusselator`, `LM-test1`, `YeastGlycolysis`, `MAPK-HF96-layout`) via `bsc.load_example()` — no `.cps` files needed. Run with `poetry run pytest`. Coverage includes
both `format_output="wide"` and `"long"` (the `auto_melt` path), and that `generate_input_space` does not mutate the caller's `scan_dict`.

Reaction parameter tests use brusselator plus `_brusselator_with_mapped_rate_constants()`, which adds test-only global parameters and maps rate constants to them, so that every dependency case (mapped to a fixed global, to one depending on a species, and through a chain of assignments) is covered.

The philosophy is "does it run and produce the expected shape/values" rather than exhaustive unit tests — add to the existing set rather than replacing it.

## Debug logging

The package logs via the standard `logging` module under the `quicksimsbio.*` namespace. For verbose output when diagnosing `auto_melt` or dependency resolution, call `logging.basicConfig(level=logging.DEBUG)` (or attach a file handler to the `quicksimsbio` logger).

## Setup & commands

```bash
poetry install                              # install dependencies
poetry run python -c "import quicksimsbio"  # smoke test
poetry run pytest                           # regression tests
```

## Known issues

Edge cases where the current code is known (or suspected) to fall short:

- **Assigned compartments in dependency resolution.** `resolve_status_df` builds its
  expression mapping from assigned _parameters_ only (`bsc.get_parameters`). An assigned
  compartment is marked `unknown` in `create_status_df` but gets no expression, so it likely
  fails in `extract_items_from_expression` (NaN input). Untested — needs a model with an
  assigned compartment to confirm.
- **ODE-type global quantities** are not classified by `create_status_df` (status stays `None`).
  Anything depending on one — including a rate constant mapped to one — cannot be resolved.
  Rare; not currently handled.
- **One unmatched reference sinks the long format.** `resolve_elements_list` raises
  `ValueError` if any element of an expression is missing from `status_df`, so a single odd
  reference stops the whole `auto_melt` / `format_output="long"` conversion. This was originally to deliberately catch unhandled reference types, but logging a warning
  and labelling as `unknown` would degrade more gracefully.

## Possible improvements

Optional ideas I didn't get time to implement:

- **`get_model_inputs` with current values** — an argument to return each input's current
  value as a single-element list instead of `None`, as a quick way to inspect the model.
- **Output names from basico rather than strings** — `_input_param_outputs` builds names like
  `f"Values[{key}]"` and `f"[{key}]_0"` by hand; using basico's `display_name` columns would
  be more robust to naming edge cases.
- **Tidy the empty-table guards** — the repeated `safe_df(...)` / `.empty` checks (e.g. in
  `get_basico_outputs`) could become a small helper.
- **More tests** — that `output_selection` is built correctly, the random-only
  `generate_input_space` path (no grid parameters), and that `get_basico_outputs` names are
  valid directly in `bsc.run_time_course_with_output`.
