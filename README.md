# quicksimsbio

A lightweight Python package that wraps [basico](https://basico.readthedocs.io/) (the Python
interface to [COPASI](http://copasi.org/)) to make ODE / systems-biology simulation, parameter
scanning, and quasi-random sampling straightforward. Models are COPASI `.cps` files, or any of
basico's bundled example models.

The public API is five functions:

| Function                                          | What it does                                                                                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `get_model_inputs(model=None)`                    | Build a `scan_dict` skeleton — every tunable parameter/species as a key, all set to `None` (meaning "hold at the model default").       |
| `run_simulations(scan_dict, timepoints, ...)`     | Run a time course over every parameter combination in a `scan_dict`.                                                                    |
| `generate_input_space(scan_dict, n_rand_samples)` | Expand a `scan_dict` into a table of parameter sets (grid × quasi-random).                                                              |
| `get_basico_outputs(model=None)`                  | List the output column names a model exposes (for `output_selection`).                                                                  |
| `fit_curve_to_data(x, y, model=...)`              | Fit a curve (logistic / hyperbola / line) to data via scipy.                                                                            |

All simulation functions can take an explicit `model=` argument. If omitted, functions fall back to
basico's currently loaded model. Passing `model=dm` explicitly is useful when working with multiple different models, and is demonstrated in the quick start examples.

## Installation

```bash
poetry install

# Smoke test
poetry run python -c "import quicksimsbio"
```

Requires Python ≥ 3.12.

## Quick start

### 1. Load a model and inspect its inputs

```python
import basico as bsc
import quicksimsbio as qsb

dm = bsc.load_example("brusselator")   # or bsc.load_model("path/to/your_model.cps")

inputs = qsb.get_model_inputs(model=dm)
# {'X': None, 'Y': None, 'A': None, 'B': None, ...}  -> every key holds at the model default
```

### 2. Run a single simulation

```python
result = qsb.run_simulations(inputs, timepoints=[0, 1, 5, 10], model=dm)
# wide DataFrame: Time, one column per transient species, and a Sim_Num column
```

### 3. Scan parameters

A `scan_dict` value can be:

- `None` — hold at the model default
- a `list` — discrete grid values
- a `dict` `{"lower": ..., "upper": ..., "log": bool}` — a quasi-random range

```python
inputs["B"] = [2.0, 3.0, 4.0]                                # 3-point grid scan over parameter B
result = qsb.run_simulations(inputs, [0, 1, 2, 3, 4, 5], model=dm)   # runs 3 simulations, from 0-5 seconds
```

Get the results in long (tidy) format for plotting with `format_output="long"`:

```python
# Long format dataframes are often more useful for plotting the results and organising data
long_df = qsb.run_simulations(inputs, [0, 5, 10], format_output="long", model=dm)
```

### 4. Build a quasi-random input space

```python
inputs = qsb.get_model_inputs(model=dm)
inputs["A"] = [1.0, 2.0]                                 # grid axis (2 values)
inputs["B"] = {"lower": 0.1, "upper": 10, "log": True}   # Quasi-random range, log-distributed points

space = qsb.generate_input_space(inputs, n_rand_samples=8)
# shape (2, 16): 2 varied params (rows) across 2 grid points x 8 samples = 16 sets (columns)
```

`method=` selects the quasi-random engine (`"sobol"` default, `"halton"`, `"lhs"`); pass
`rng=<seed>` for reproducibility.

### 5. Discover output column names

```python
outputs = qsb.get_basico_outputs(model=dm)   # DataFrame describing every available output for that model
species = outputs[outputs["Type"] == "Species Concentration"]["Transient"].tolist()
result = qsb.run_simulations(inputs, [0, 5, 10], output_selection=species, model=dm)
```

### 6. Fit a curve to data

```python
import numpy as np
import pandas as pd

x = pd.Series(np.linspace(0.01, 100, 30))
y = pd.Series(10 / (1 + (5 / x) ** 2))

fit = qsb.fit_curve_to_data(x, y, model="3 parameter logistic")
# {'Emax': ..., 'Emin': ..., 'EC50': ..., 'Emax_stdev': ..., 'Cov(Emax, Emin)': ..., ...}
```

Supported models: `"3 parameter logistic"`, `"4 parameter logistic"`,
`"rectangular_hyperbola"`, `"straight_line"`. Pass `x`/`y` as pandas Series.

## Output column naming (basico conventions)

| Variable type                   | `output_selection` format |
| ------------------------------- | ------------------------- |
| Transient species concentration | `[SpeciesName]`           |
| Initial species concentration   | `[SpeciesName]_0`         |
| Fixed parameter (transient)     | `Values[ParamName]`       |
| Reaction flux                   | `(ReactionName).Flux`     |
| Time                            | `Time`                    |

## Development

```bash
poetry install
poetry run pytest        # regression tests against bundled example models
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the package's design decisions, internal architecture,
and maintainer notes — start here if you're inheriting or extending the codebase.

## License

MIT — see [LICENSE](LICENSE).
