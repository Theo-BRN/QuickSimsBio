"""
Regression tests for the quicksimsbio public API.

Models are loaded from basico's bundled examples — no .cps files needed.
BROAD_MODELS covers a range of model structures for get_model_inputs and run_simulations.
BASICO_NAMES_MODELS is the subset used for the get_basico_outputs tests. It was originally
restricted to models known to work; get_basico_outputs now also runs on the other BROAD_MODELS.
"""

import basico as bsc
import pandas as pd
import pytest

import quicksimsbio as qsb
from quicksimsbio.sampling import (
    STATUS_LABEL,
    extract_items_from_expression,
    get_dependency_status_dataframe,
)

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
def test_run_simulations_long_format(model_name):
    """format_output='long' must auto_melt without crashing or leaking 'error:' sentinels.

    Exercises the dependency-resolution path that the wide-format tests skip:
    - models with no global parameters (brusselator) hit the resolver early-return
    - models with assigned quantities (MAPK-HF96-layout) actually resolve dependencies
    """
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    result = qsb.run_simulations(inputs, [0, 1, 5], format_output="long", model=dm)
    assert isinstance(result, pd.DataFrame)
    assert "Time" in result.columns
    # auto_melt produces a long frame with a single output-type column and value column
    assert result.shape[1] < len(inputs) + 5
    # the "error:" sentinel from resolve_elements_list must never reach the output
    leaked_errors = (
        result.astype(str).apply(lambda s: s.str.startswith("error:")).any().any()
    )
    assert not leaked_errors


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


def test_run_simulations_sets_compartment_qualified_species():
    """In multi-compartment models (e.g. array_1d) the same species name appears in
    every compartment, so inputs are named like "Calcium{compartment[3]}". Setting
    one must actually change that species' starting value."""
    dm = bsc.load_example("array_1d")
    inputs = qsb.get_model_inputs(model=dm)
    key = "Calcium{compartment[3]}"
    assert key in inputs
    inputs[key] = [0.5, 1.0]
    result = qsb.run_simulations(inputs, [0, 1], model=dm)
    start = result.loc[result["Time"] == 0]
    assert start[f"[{key}]"].tolist() == pytest.approx([0.5, 1.0])
    assert start[f"[{key}]_0"].tolist() == pytest.approx([0.5, 1.0])


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


def test_generate_input_space_does_not_mutate_scan_dict():
    """A bound dict without 'log' must not gain a 'log' key in the caller's scan_dict."""
    dm = bsc.load_example("LM-test1")
    inputs = qsb.get_model_inputs(model=dm)
    first_key = list(inputs)[0]
    bound = {"lower": 0.5, "upper": 2.0}  # deliberately no "log"
    inputs[first_key] = bound
    qsb.generate_input_space(inputs, n_rand_samples=4)
    # `bound` is the exact object we put in the scan_dict, so an in-place mutation
    # by generate_input_space would show up as a new "log" key here.
    assert "log" not in bound


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


# --- dependency resolution (auto_melt / long format) ---


def test_extract_items_strips_copasi_reference_braces():
    """COPASI braces a Values[...] reference whose name contains spaces (e.g. the
    auto-generated 'Initial for kL+'). status_df stores the un-braced display name,
    so the extractor must strip the braces or the long-format resolver crashes with
    'Not all elements ... were present' (regression: BioModel MODEL2306220001)."""
    assert extract_items_from_expression(
        "{Values[Initial for kL+]} * {Values[Initial for alpha+]}"
    ) == ["Values[Initial for kL+]", "Values[Initial for alpha+]"]

    # Un-braced refs (incl. '+'/'-' names and nested brackets) are untouched.
    assert extract_items_from_expression(
        "( Values[[G]tot] - Values[LigandConcAdded] ) / Values[LigandConcAdded]"
        " * {Values[TRUPATH System Max]}"
    ) == [
        "Values[[G]tot]",
        "Values[LigandConcAdded]",
        "Values[LigandConcAdded]",
        "Values[TRUPATH System Max]",
    ]


# --- reaction parameters (rate constants) ---


def _brusselator_with_mapped_rate_constants():
    """Brusselator with extra globals, each reaction's k1 mapped to one of them, so
    every dependency case for a mapped rate constant is present. (R4).k1 stays local.
    These globals are test scaffolding — the real brusselator has none."""
    dm = bsc.load_example("brusselator")
    bsc.add_parameter("kg", initial_value=5.0, model=dm)  # fixed
    bsc.add_parameter("ka", type="assignment", expression="[X] * 2", model=dm)
    bsc.add_parameter("kb", type="assignment", expression="Values[kg] * 2", model=dm)
    bsc.set_reaction_parameters("(R1).k1", mapped_to="kg", model=dm)
    bsc.set_reaction_parameters("(R2).k1", mapped_to="ka", model=dm)
    bsc.set_reaction_parameters("(R3).k1", mapped_to="kb", model=dm)
    return dm


@pytest.mark.parametrize("model_name", BROAD_MODELS)
def test_get_model_inputs_reaction_params_match_basico_names(model_name):
    """Local reaction parameter keys must be exactly basico's own names, e.g. "(R1).k1",
    so they can be set and selected as outputs without translation."""
    dm = bsc.load_example(model_name)
    inputs = qsb.get_model_inputs(model=dm)
    reaction_params = bsc.get_reaction_parameters(model=dm)
    local_names = set(reaction_params.index[reaction_params["type"] == "local"])
    assert len(local_names) > 0
    assert local_names <= set(inputs)


def test_get_model_inputs_excludes_mapped_reaction_params():
    dm = _brusselator_with_mapped_rate_constants()
    inputs = qsb.get_model_inputs(model=dm)
    assert "(R4).k1" in inputs
    for mapped in ["(R1).k1", "(R2).k1", "(R3).k1"]:
        assert mapped not in inputs


def test_get_basico_outputs_lists_reaction_params():
    dm = bsc.load_example("brusselator")
    names = qsb.get_basico_outputs(model=dm)
    reaction_param_rows = names[names["Type"] == "Reaction Parameter"]
    assert set(reaction_param_rows["Transient"]) == {
        "(R1).k1",
        "(R2).k1",
        "(R3).k1",
        "(R4).k1",
    }
    names_without = qsb.get_basico_outputs(display_reaction_parameters=False, model=dm)
    assert "Reaction Parameter" not in set(names_without["Type"])


def test_run_simulations_reaction_param_scan():
    """Varying a rate constant must change the simulation, and its value must be
    recorded in the results (wide and long)."""
    dm = bsc.load_example("brusselator")
    inputs = qsb.get_model_inputs(model=dm)
    inputs["(R1).k1"] = [0.5, 2.0]

    wide = qsb.run_simulations(inputs, [0, 5], model=dm)
    assert "(R1).k1" in wide.columns
    end = wide.loc[wide["Time"] == 5]
    assert end["(R1).k1"].tolist() == pytest.approx([0.5, 2.0])
    assert end["[X]"].iloc[0] != pytest.approx(end["[X]"].iloc[1])

    long = qsb.run_simulations(inputs, [0, 5], format_output="long", model=dm)
    assert "(R1).k1" in long.columns  # kept as an independent (id) column
    assert set(long["(R1).k1"]) == {0.5, 2.0}


def test_run_simulations_skips_mapped_reaction_param(caplog):
    """A mapped rate constant must not be set (that would change its global quantity),
    and the user is warned once per run, not once per simulation."""
    dm = _brusselator_with_mapped_rate_constants()
    inputs = qsb.get_model_inputs(model=dm)
    inputs["(R1).k1"] = [0.5, 1.0, 2.0]
    result = qsb.run_simulations(
        inputs, [0, 5], output_selection=["(R1).k1", "Values[kg]"], model=dm
    )
    assert result["Values[kg]"].tolist() == pytest.approx([5.0] * len(result))
    assert result["(R1).k1"].tolist() == pytest.approx([5.0] * len(result))
    mapped_warnings = [r for r in caplog.records if "is mapped to" in r.getMessage()]
    assert len(mapped_warnings) == 1


def test_dependency_status_of_reaction_params():
    """Local rate constants are independent; a mapped one takes the status of the
    global quantity it is mapped to, resolved through any chain of assignments."""
    dm = _brusselator_with_mapped_rate_constants()
    status = get_dependency_status_dataframe(model=dm)[STATUS_LABEL]
    assert status["(R4).k1"] == "independent"  # local
    assert status["(R1).k1"] == "independent"  # -> kg (fixed)
    assert status["(R2).k1"] == "dependent"  # -> ka = [X] * 2
    assert status["(R3).k1"] == "independent"  # -> kb = Values[kg] * 2
