# QuickSimsBio

A lightweight Python package wrapped around [basico](https://basico.readthedocs.io/) (the Python interface to [COPASI](http://copasi.org/)). QuickSimsBio provides convenient functions for the process of running mechanistic model / systems biology simulations and adds some additional functionality like quasi-random sampling, curve fitting and automatic dataframe melting to a long format. It should be used alongside basico for convenience for certain functions.

The public API is five functions:

| Function                                          | What it does                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `get_model_inputs(model=None)`                    | Build a scan dictionary `scan_dict` skeleton — every tunable parameter/species/rate constant is a key. |
| `generate_input_space(scan_dict, n_rand_samples)` | Expand a `scan_dict` into a table of parameter sets (grid × quasi-random bounds).                      |
| `run_simulations(scan_dict, timepoints, ...)`     | Run a time course simulation for each parameter combination in a `scan_dict`.                          |
| `get_basico_outputs(model=None)`                  | Display the possible output column names for a given model (can be used in `output_selection`).        |
| `fit_curve_to_data(x, y, model=...)`              | Fit a curve (logistic / hyperbola / line) to data via scipy.                                           |

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

dm = bsc.load_example("brusselator")
# or bsc.load_model("path/to/your_model.cps")

inputs = qsb.get_model_inputs(model=dm)
# {'X': None, 'Y': None, 'A': None, 'B': None, ..., '(R1).k1': None, ...}
# -> None means no change to the input, hence uses current value
```

Inputs are the model's fixed global parameters, its species (initial concentrations), and its
reaction (local) parameters such as rate constants. Reaction parameters use basico's
`(ReactionName).ParamName` naming, e.g. `(R1).k1`.

### 2. Run a single simulation

```python
result = qsb.run_simulations(inputs, timepoints=[0, 1, 5, 10], model=dm)
# wide DataFrame: Time, one column per transient species, and a Sim_Num column
```

### 3. Scan parameters

A `scan_dict` value can be:

- `None` — hold at the current value
- a `list` — discrete grid values
- a `dict` `{"lower": ..., "upper": ..., "log": bool}` — a quasi-random range

```python
inputs["B"] = [2.0, 3.0, 4.0] # 3-point grid scan over the initial concentration of species B
result = qsb.run_simulations(inputs, [0, 1, 2, 3, 4, 5], model=dm)   # runs 3 simulations, from 0-5 seconds
```

Reaction rate constants are scanned the same way:

```python
inputs = qsb.get_model_inputs(model=dm)
inputs["(R1).k1"] = [0.5, 1.0, 2.0] # 3-point grid scan over rate constant k1 of reaction R1
result = qsb.run_simulations(inputs, [0, 5, 10], model=dm)
```

A rate constant that is _mapped_ to a global quantity (in COPASI, the reaction uses a global
parameter instead of its own local value) is not listed by `get_model_inputs` and is not changed
if supplied — vary the global parameter instead. A warning is logged if you try.

Get the results in a nicer (long) format for plotting with `format_output="long"`:

```python
# Long format dataframes are often more useful for plotting the results and organising data
long_df = qsb.run_simulations(inputs, [0, 5, 10], format_output="long", model=dm)
```

### 4. Build a quasi-random input space

```python
inputs = qsb.get_model_inputs(model=dm)
inputs["A"] = [1.0, 2.0] # grid axis (2 values)
inputs["B"] = {"lower": 0.1, "upper": 10, "log": True} # Quasi-random range, log-distributed points

space = qsb.generate_input_space(inputs, n_rand_samples=8)
# shape (2, 16): 2 varied params (rows) across 2 grid points x 8 samples = 16 sets (columns)
```

`method=` selects the quasi-random engine (`"sobol"` default, `"halton"`, `"lhs"`); pass an `rng=<seed>` for reproducibility.

### 5. Discover output column names

```python
outputs = qsb.get_basico_outputs(model=dm) # DataFrame describing every available output for that model
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

## Development

```bash
poetry install
poetry run pytest  # regression tests against bundled example models
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the package's design decisions, internal architecture,
and maintainer notes — start here to contribute / extend the codebase.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Development of QuickSimsBio began during T. Redfern-Nichols's PhD in the Ladds Lab (Department of Pharmacology, University of Cambridge). It has since been developed further at the University of Liverpool.
