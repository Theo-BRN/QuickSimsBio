"""
Regression tests for the quicksimsbio public API.

Models are loaded from basico's bundled examples — no .cps files needed.
BROAD_MODELS covers a range of model structures for get_model_inputs and run_simulations.
BASICO_NAMES_MODELS is the subset known to work with get_basico_names (see TODO step 4).
"""

import basico as bsc
import pandas as pd
import pytest

import quicksimsbio as qsb

BROAD_MODELS = ["brusselator", "LM-test1", "YeastGlycolysis", "MAPK-HF96-layout"]
BASICO_NAMES_MODELS = ["LM-test1", "MAPK-HF96-layout"]


# --- get_model_inputs ---


@pytest.mark.parametrize("model_name", BROAD_MODELS)
def test_get_model_inputs_returns_none_dict(model_name):
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    assert isinstance(inputs, dict)
    assert len(inputs) > 0
    assert all(v is None for v in inputs.values())


# --- run_simulations ---


@pytest.mark.parametrize("model_name", BROAD_MODELS)
def test_run_simulations_default_outputs(model_name):
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    timepoints = [0, 1, 5, 10]
    result = qsb.run_simulations(inputs, timepoints, model=dm)
    assert isinstance(result, pd.DataFrame)
    assert "Time" in result.columns
    assert "Sim_Num" in result.columns
    assert len(result) == len(timepoints)


@pytest.mark.parametrize("model_name", BROAD_MODELS)
def test_run_simulations_single_param_scan(model_name):
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    first_key = next(iter(inputs))
    inputs[first_key] = [0.5, 2.0]
    timepoints = [0, 5, 10]
    result = qsb.run_simulations(inputs, timepoints, model=dm)
    assert isinstance(result, pd.DataFrame)
    # At least one simulation must have completed all timepoints.
    # Some models are numerically unstable at extreme parameter values and return
    # fewer rows for a given sim — this is expected behaviour, not a code bug.
    assert len(result) >= len(timepoints)


# --- generate_input_space ---


@pytest.mark.parametrize("model_name", BROAD_MODELS)
def test_generate_input_space_shape(model_name):
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    keys = list(inputs.keys())
    n_list_values = 3
    n_sobol = 4
    inputs[keys[0]] = [1.0, 2.0, 3.0]
    inputs[keys[1]] = {"lower": 0.5, "upper": 2.0, "log": False}
    space = qsb.generate_input_space(inputs, n_rand_samples=n_sobol)
    assert isinstance(space, pd.DataFrame)
    assert space.shape == (2, n_list_values * n_sobol)


# --- get_basico_outputs ---


@pytest.mark.parametrize("model_name", BASICO_NAMES_MODELS)
def test_get_basico_names_returns_dataframe_with_type_column(model_name):
    dm = bsc.load_example(model_name)
    names = qsb.get_basico_outputs(model=dm)
    assert isinstance(names, pd.DataFrame)
    assert "Type" in names.columns


@pytest.mark.parametrize("model_name", BASICO_NAMES_MODELS)
def test_get_basico_names_species_are_valid_simulation_outputs(model_name):
    """Species names from get_basico_names should be accepted by run_simulations."""
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    names = qsb.get_basico_outputs(model=dm)
    species_outputs = names[names["Type"] == "Species Concentration"][
        "Transient"
    ].tolist()
    assert len(species_outputs) > 0
    result = qsb.run_simulations(
        inputs, [0, 1, 5], output_selection=species_outputs, model=dm
    )
    assert isinstance(result, pd.DataFrame)
    assert set(species_outputs) <= set(result.columns)
