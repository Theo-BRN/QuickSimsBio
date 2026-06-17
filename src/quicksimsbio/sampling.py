import itertools
import logging
from typing import Any, Optional

import basico as bsc
import numpy as np
import numpy.typing
import pandas as pd
from scipy.stats import qmc

logger = logging.getLogger(__name__)

TIME_SECONDS_COLUMN = "Time (seconds)"
TIME_MINUTES_COLUMN = "Time (minutes)"
OUTPUT_VALUE_NAME = "Model_Output"
OUTPUT_TYPE_NAME = "Model_Output_Type"
STATUS_LABEL = "dependency_status"


def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Ensures a DataFrame is returned even if the source returns None."""
    return pd.DataFrame() if df is None else df


ScanValue = (
    None
    | list[float | int]
    | dict[str, bool | float]
    | numpy.typing.NDArray[np.floating[Any] | np.integer[Any]]
)


def get_model_inputs(model=None) -> dict[str, ScanValue]:
    """
    This is a working replacement of create_parameter_scan.
    Instead the dictionary will be created from a temporarily created basico parameter_set
    """
    # TODO add argument to this to also provide the default/currently loaded values in single element lists instead of None
    # (would be a quick way of seeing the current parameters etc)

    # Should work off given model or globally loaded model
    dm = model if model is not None else bsc.get_current_model()

    # Create temporary parameter set name
    new_parameter_set_name = "temporary"
    p_sets = bsc.get_parameter_sets(model=dm)
    p_set_names = [n["name"] for n in p_sets]
    limit = 0
    while new_parameter_set_name in p_set_names:
        logger.info(
            f"temporary parameter set name '{new_parameter_set_name}' already exists, trying next"
        )
        new_parameter_set_name = f"{new_parameter_set_name} copy"
        limit += 1
        if limit == 100:
            raise ValueError(
                "Something is probably going wrong with the temporary parameter set name"
            )

    # extract inputs from parameter set dictionary
    bsc.add_parameter_set(new_parameter_set_name, model=dm)
    extracted_p_sets = bsc.get_parameter_sets(
        new_parameter_set_name, exact=True, model=dm
    )
    bsc.remove_parameter_sets(new_parameter_set_name, exact=True, model=dm)

    if len(extracted_p_sets) > 1:
        raise ValueError(
            "Somehow extracted multiple parameter sets when trying to provide model inputs"
        )

    param_set = extracted_p_sets[0]
    ignored_keys = ["name", "description", "Initial Time", "Initial Compartment Sizes"]

    for k in ignored_keys:
        param_set.pop(k)

    inputs = []
    species = param_set["Initial Species Values"].keys()
    input_params = [
        p
        for p, v in param_set["Initial Global Quantities"].items()
        if v["simulation_type"] == "fixed"
    ]
    inputs.extend(input_params)
    inputs.extend(species)

    scan_dict: dict[str, ScanValue] = {str(k): None for k in inputs}

    return scan_dict


def default_timepoints():
    logger.warning("No timepoints set — defaulting to 0 → 1000 s (n=300)")
    return np.linspace(0, 1000, 300).tolist()


def _default_species_outputs(model, conc_or_number="Concentration") -> list[str]:
    """Returns transient species output names for all species in the model."""
    outputs = get_basico_outputs(
        display_parameters=False,
        display_species=True,
        display_reactions=False,
        display_compartments=False,
        model=model,
    )
    if outputs.empty:
        return []
    elif conc_or_number == "Concentration":
        return outputs.loc[
            outputs["Type"] == "Species Concentration", "Transient"
        ].to_list()
    elif conc_or_number == "Particle":
        return outputs.loc[outputs["Type"] == "Species Particle", "Transient"].to_list()
    else:
        raise ValueError(
            "Please supply 'Concentration' or 'Particle' for default species output"
        )


def _input_param_outputs(varying_params, model) -> list[str]:
    """Returns basico output names for the parameters/species that were varied,
    so results DataFrames record what changed in each simulation."""
    params = safe_df(bsc.get_parameters(model=model))
    set_params = set([] if params.empty else params.index)
    species_df = safe_df(bsc.get_species(model=model))
    set_species = set(species_df.index)

    # TODO Replace this with display name column from params_df instead of string manipulation
    output_params = [f"Values[{key}]" for key in varying_params if key in set_params]
    output_species = [f"[{key}]_0" for key in varying_params if key in set_species]

    unrecognised = [
        key for key in varying_params if key not in set_params | set_species
    ]
    if unrecognised:
        logger.warning(
            f"scan_dict items {unrecognised} were not recognised as species or parameters"
        )

    return output_params + output_species


def _build_output_selection(
    model,
    output_selection,
    varying_params,
    include_input_params,
    include_time,
):
    """Resolve the full output_selection list from the user's intent.

    When output_selection is None, defaults to all transient species concentrations.
    In this case, include_input_params=None is promoted to True so the results
    DataFrame automatically records which parameter values produced each simulation.
    When output_selection is provided, include_input_params stays None (falsy) and
    nothing is appended unless the caller explicitly passes include_input_params=True.
    Time is always prepended as the first column unless include_time=False.
    """
    if output_selection is None:
        output_selection = _default_species_outputs(model=model)

        if include_input_params is None:
            include_input_params = True
    else:
        if not isinstance(output_selection, list):
            raise ValueError("Please supply output selection as a list")

    # Add input params
    if include_input_params:
        output_selection = output_selection + _input_param_outputs(
            varying_params, model=model
        )

    # User must set include_time as False to drop Time column
    if include_time is not False:
        output_selection = ["Time"] + output_selection

    return output_selection


def run_simulations(
    scan_dict,
    timepoints=None,
    output_selection=None,
    format_output="wide",
    *,
    include_input_params=None,
    include_time=None,
    n_rand_samples=None,
    model: bsc.COPASI.CDataModel | None = None,
    **kwargs,
):
    """Run time-course simulations over all parameter combinations in scan_dict.

    Args:
        scan_dict: Parameter combinations to iterate over (from get_model_inputs).
        timepoints: List of time values to simulate. Defaults to 0–1000 s (n=300).
        output_selection: Explicit list of basico output names. When None, defaults
            to all transient species.
        format_output: 'wide' (default) or 'long' (via auto_melt).
        include_input_params: True = always append varied-parameter columns to output.
            None (default) = append only when no explicit output_selection was given.
            False = never append.
        include_time: True = always prepend Time column. None (default) = prepend only
            when no explicit output_selection was given. False = never prepend.
        n_rand_samples: Required when scan_dict contains Sobol bound dicts.
        model: COPASI model. Defaults to the globally loaded model.
    """
    dm = model if model is not None else bsc.get_current_model()
    timepoints = timepoints if timepoints is not None else default_timepoints()

    # Extract parameters that need to be varied from the scan_dict
    varying_params = {
        key: value for key, value in scan_dict.items() if value is not None
    }
    param_sets_df = generate_input_space(varying_params, n_rand_samples)

    output_selection = _build_output_selection(
        model=dm,
        output_selection=output_selection,
        varying_params=varying_params,
        include_input_params=include_input_params,
        include_time=include_time,
    )

    # Iterate over each parameter set
    sim_df_list = []
    for sim_num, (param_set_name, param_set) in enumerate(
        param_sets_df.items(), start=1
    ):
        sim_df = run_individual_simulation_for_param_set(
            param_set=param_set,
            timepoints=timepoints,
            output_selection=output_selection,
            sim_num=sim_num,
            model=dm,
            **kwargs,
        )
        sim_df_list.append(sim_df)

    # Concatenate all individual simulation dataframes
    sim_df_all = pd.concat(sim_df_list, ignore_index=True)

    if format_output == "wide":
        return sim_df_all
    elif format_output != "long":
        raise ValueError(
            f"format_output must be 'wide' or 'long', got {format_output!r}"
        )
    else:
        return auto_melt(sim_df_all, model=dm)


def auto_melt(
    simulation_df,
    var_name=OUTPUT_TYPE_NAME,
    value_name=OUTPUT_VALUE_NAME,
    *,
    model: bsc.COPASI.CDataModel,
):
    logger.debug(f"attempting to auto_melt df with columns:\n{simulation_df.columns}")
    vars_dict = get_dependency_dictionary(model=model)
    independent_vars, dependent_vars, remaining_unknown_vars = vars_dict.values()
    logger.debug(
        f"dependency dictionary had "
        f"\n{len(independent_vars)} independent_vars:\n{independent_vars}, "
        f"\n{len(dependent_vars)} dependent_vars:\n{dependent_vars} and "
        f"\n{len(remaining_unknown_vars)} remaining_unknown_vars:\n{remaining_unknown_vars}"
    )

    # checking for duplicate independent_vars columns (else pd.melt breaks)
    duplicate_col_names = simulation_df.columns[
        simulation_df.columns.duplicated()
    ].unique()
    if len(duplicate_col_names) > 0:
        # removing duplicates if possible
        for col in set(duplicate_col_names):
            duplicates = simulation_df.loc[:, simulation_df.columns == col]
            if (duplicates.nunique(axis=1) == 1).all():
                logger.debug(
                    f"While trying to auto_melt a dataframe, the column {col} appeared more than once, but with identical data, duplicates will be removed."
                )
            else:
                raise ValueError(
                    f"While trying to auto_melt a dataframe, the column {col} appeared more than once, containing different data, unsure which columns to remove."
                )
        # reset df to not include duplicate cols
        simulation_df = simulation_df.loc[:, ~simulation_df.columns.duplicated()].copy()

    # remove duplicates (else pd.melt breaks), while preserving order (preference)
    independent_vars = list(dict.fromkeys(independent_vars))

    # the supplied id_vars and value_vars must be present in the df or else pd.melt breaks
    independent_vars = [
        indep_var
        for indep_var in independent_vars
        if indep_var in simulation_df.columns
    ]
    dependent_vars = [
        dep_var for dep_var in dependent_vars if dep_var in simulation_df.columns
    ]

    # Melt dataframe
    simulation_df = pd.melt(
        simulation_df,
        id_vars=independent_vars,
        value_vars=dependent_vars,
        var_name=var_name,
        value_name=value_name,
    )

    return simulation_df


def _generate_parameter_sets_df(scan_dict):
    """
    Create a dataframe with columns corresponding to different parameter sets derived from the scan_dict.

    Each column in the returned dataframe represents a unique combination of parameters from the scan_dict.

    Parameters:
    - scan_dict (dict): A dictionary where keys are parameter names and values are lists of values to scan over.

    Returns:
    - pd.DataFrame: A dataframe where each column represents a unique parameter set.
    """
    # Get the parameter values lists (param_values is a list of lists)
    param_values = list(scan_dict.values())
    param_combinations = []

    # Generate all possible combinations using the parameter values lists
    for combination in itertools.product(*param_values):
        param_combinations.append(combination)

    # Convert the combinations into a dataframe
    df = pd.DataFrame(param_combinations, columns=scan_dict.keys()).T

    # Rename columns to "Parameter Set i"
    df.columns = [f"Parameter Set {i + 1}" for i in range(df.shape[1])]
    return df


def _sample_quasi_random(
    df_bounds: pd.DataFrame,
    n_samples: int,
    method: str = "sobol",
    rng=None,
) -> pd.DataFrame:
    """Draw quasi-random samples scaled to parameter bounds.

    Args:
        df_bounds: DataFrame with index = parameter names, columns ["lower", "upper"].
        n_samples: Number of samples to draw.
        method: QMC engine to use. One of "sobol", "halton", "lhs".
        rng: Seed or numpy Generator for reproducibility. None = random each run.
    """
    d = len(df_bounds)
    match method:
        case "sobol":
            engine = qmc.Sobol(d=d, scramble=True, rng=rng)
        case "halton":
            engine = qmc.Halton(d=d, scramble=True, rng=rng)
        case "lhs":
            engine = qmc.LatinHypercube(d=d, rng=rng)
        case _:
            raise ValueError(
                f"Unknown sampling method {method!r}. Choose 'sobol', 'halton', or 'lhs'."
            )
    sample = engine.random(n=n_samples)
    scaled = qmc.scale(
        sample, df_bounds["lower"].to_numpy(), df_bounds["upper"].to_numpy()
    )
    return pd.DataFrame(scaled, columns=list(df_bounds.index))


def categorise_scan_dict(scan_dict):
    individual_params = {}
    random_params = {}
    ranged_params = {}

    for key, scan_dict_value in scan_dict.items():
        if scan_dict_value is None:
            continue
        elif isinstance(scan_dict_value, dict):
            required_inputs = {"upper", "lower"}
            if required_inputs <= scan_dict_value.keys():
                random_params[key] = scan_dict_value
            else:
                missing_bound = [
                    e for e in required_inputs if e not in scan_dict_value.keys()
                ]
                raise ValueError(
                    f"Could not find {missing_bound} in scan dictionary's '{key}' key"
                )
            if "log" not in scan_dict_value.keys():
                random_params[key]["log"] = False
        elif isinstance(scan_dict_value, (list, np.ndarray)):
            if len(scan_dict_value) == 1:
                individual_params[key] = scan_dict_value
            elif len(scan_dict_value) > 1:
                ranged_params[key] = scan_dict_value
            else:
                raise ValueError(f"Empty list for key '{key}' not allowed.")
        else:
            raise ValueError(
                f"Unexpected value type for key '{key}': {type(scan_dict_value)}"
            )

    accounted_for = {**individual_params, **random_params, **ranged_params}
    missing = {
        k: v for k, v in scan_dict.items() if v is not None and k not in accounted_for
    }
    if missing:
        raise ValueError(f"Unaccounted-for parameters: {list(missing)}")

    return {
        "individual_params": individual_params,
        "random_params": random_params,
        "ranged_params": ranged_params,
    }


def combine_param_sets(df1, df2):
    combined_dfs = []
    combined_names = []
    for col1, col2 in itertools.product(df1.columns, df2.columns):
        combined = pd.concat([df1[col1], df2[col2]])
        name = f"{col1}_{col2}"
        combined_dfs.append(combined)
        combined_names.append(name)
    df_combined = pd.concat(combined_dfs, axis=1)
    df_combined.columns = combined_names
    return df_combined


def generate_input_space(
    scan_dict,
    n_rand_samples=None,
    method: str = "sobol",
    rng=None,
):
    """Generate a parameter space from any scan_dict, handling both grid and quasi-random params.

    Args:
        scan_dict: Dict of parameter names to ScanValue. None = hold at default,
            list = discrete grid values, dict with "lower"/"upper"/"log" = random range.
        n_rand_samples: Number of quasi-random samples to draw per grid point.
            Required when scan_dict contains quasi-random bound dicts; ignored otherwise.
        method: QMC engine. One of "sobol" (default), "halton", "lhs".
        rng: Seed or numpy Generator for reproducibility. None = random each run.
    """
    cats_dict = categorise_scan_dict(scan_dict)
    rand_dict = cats_dict["random_params"]

    # Grid-only path: no quasi-random params present
    if not rand_dict:
        grid_params = {**cats_dict["individual_params"], **cats_dict["ranged_params"]}
        return _generate_parameter_sets_df(grid_params)

    # Quasi-random path
    if n_rand_samples is None:
        raise ValueError(
            "scan_dict contains quasi-random bound dicts but n_rand_samples was not provided. "
            "Pass n_rand_samples=<int> to generate_input_space."
        )

    # Build bounds using original param names throughout.
    log_scaled_params: list[str] = []
    bounds: dict[str, dict[str, float]] = {}
    for param_key, param_value in rand_dict.items():
        if param_value["log"]:
            bounds[param_key] = {
                "lower": np.log10(param_value["lower"]),
                "upper": np.log10(param_value["upper"]),
            }
            log_scaled_params.append(param_key)
        else:
            bounds[param_key] = {
                "lower": param_value["lower"],
                "upper": param_value["upper"],
            }

    df_bounds = pd.DataFrame(bounds).T
    quasi_samples = _sample_quasi_random(
        df_bounds, n_rand_samples, method=method, rng=rng
    )

    for param_key in log_scaled_params:
        quasi_samples[param_key] = 10.0 ** quasi_samples[param_key]

    grid_samples = _generate_parameter_sets_df(
        {**cats_dict["individual_params"], **cats_dict["ranged_params"]}
    )
    return combine_param_sets(grid_samples, quasi_samples.T)


def run_individual_simulation_for_param_set(
    param_set,
    timepoints,
    output_selection,
    specified_FGQs=None,
    sim_num=None,
    *,
    model: bsc.COPASI.CDataModel | None = None,
    **kwargs,
):
    """
    This is a new version of run_simulation_for_param_set
    """
    dm = model if model is not None else bsc.get_current_model()
    assert isinstance(dm, bsc.COPASI.CDataModel)

    set_model_parameters(param_set, model=dm)

    # Run simulations with potential kwargs
    sim_df = bsc.run_time_course_with_output(
        values=timepoints,
        output_selection=output_selection,
        start_time=timepoints[0],
        model=dm,
        **kwargs,
    )

    # Add simulation number
    if sim_num is not None:
        sim_df["Sim_Num"] = sim_num

    return sim_df


def get_dependency_dictionary(*, model: bsc.COPASI.CDataModel):
    status_df = get_dependency_status_dataframe(model=model)
    dependency_labels = ["independent", "dependent", "unknown"]
    dep_dict = {}
    for each_label in dependency_labels:
        dep_dict[each_label] = status_df.index[status_df[STATUS_LABEL] == each_label]

    return dep_dict


def get_dependency_status_dataframe(*, model: bsc.COPASI.CDataModel):
    status_df = create_status_df(model=model)
    status_df = resolve_status_df(status_df, model=model)
    return status_df


def create_status_df(*, model: bsc.COPASI.CDataModel):
    status_df = initialise_status_df(model=model)
    status_df.loc[
        (status_df["Type"] == "Species Concentration")
        & (status_df["Descriptor"] == "Initial"),
        STATUS_LABEL,
    ] = "independent"
    status_df.loc[
        (status_df["Type"] == "Species Concentration")
        & (status_df["Descriptor"] != "Initial"),
        STATUS_LABEL,
    ] = "dependent"
    status_df.loc[
        (status_df["Type"] == "Species Particle")
        & (status_df["Descriptor"] == "Initial"),
        STATUS_LABEL,
    ] = "independent"
    status_df.loc[
        (status_df["Type"] == "Species Particle")
        & (status_df["Descriptor"] != "Initial"),
        STATUS_LABEL,
    ] = "dependent"
    status_df.loc[status_df["Type"] == "Reaction", STATUS_LABEL] = "dependent"

    # Parameters can be fixed or assigned, determining some variable dependency
    params_df = safe_df(bsc.get_parameters(model=model))
    if not params_df.empty:
        # Collate fixed or assigned params
        fixed_params = params_df.loc[params_df["type"] == "fixed", "display_name"]
        assigned_params = params_df.loc[
            params_df["type"] == "assignment", "display_name"
        ]

        # Create bool series
        is_fixed_param = (status_df["Type"] == "Parameter") & (
            status_df["Core_Constituent"].isin(fixed_params)
        )
        is_assigned_param = (status_df["Type"] == "Parameter") & (
            status_df["Core_Constituent"].isin(assigned_params)
        )

        # Update labelling and dependency where possible
        status_df.loc[is_fixed_param, ["Type"]] = "Fixed Parameter"
        status_df.loc[is_fixed_param, STATUS_LABEL] = "independent"
        status_df.loc[is_assigned_param, ["Type"]] = "Assigned Parameter"
        status_df.loc[is_assigned_param, STATUS_LABEL] = "unknown"

    # The Compartment Volume can be fixed or assigned
    all_comps = safe_df(bsc.get_compartments())
    fixed_comps = all_comps.loc[all_comps["type"] == "fixed", "display_name"]
    is_fixed_comp = (status_df["Type"] == "Compartment") & (
        status_df["Core_Constituent"].isin(fixed_comps)
    )
    status_df.loc[is_fixed_comp, ["Type"]] = "Fixed Compartment"
    status_df.loc[is_fixed_comp, STATUS_LABEL] = "independent"
    assigned_comps = all_comps.loc[all_comps["type"] == "assignment", "display_name"]
    is_assigned_comp = (status_df["Type"] == "Compartment") & (
        status_df["Core_Constituent"].isin(assigned_comps)
    )
    status_df.loc[is_assigned_comp, ["Type"]] = "Assigned Compartment"
    status_df.loc[is_assigned_comp, STATUS_LABEL] = "unknown"

    return status_df


def resolve_status_df(status_df, *, model: bsc.COPASI.CDataModel):
    """
    Function takes in a status_df made by `create_status_df()` and resolves any unknown dependencies.
    These unknown dependencies arise from assigned parameters or compartments.
    """

    # Generating expressions mapping
    param_df = safe_df(bsc.get_parameters(model=model))
    agq_expressions = param_df[param_df["type"] == "assignment"].set_index(
        "display_name"
    )["expression"]
    logger.info(f"agq_expressions is:\n{agq_expressions}\n\n")

    # Generating the 'unresolved_df'
    unresolved_df = status_df.loc[status_df[STATUS_LABEL] == "unknown", :].copy()
    logger.info(f"unresolved_df is:\n{unresolved_df}")

    # Combining expressions mapping with 'unresolved_df'
    unresolved_df["expression"] = unresolved_df["Core_Constituent"].map(agq_expressions)
    logger.info(f"unresolved_df with mapped expressions is:\n{unresolved_df}")

    unresolved_df["constituent_elements"] = unresolved_df["expression"].map(
        extract_items_from_expression
    )

    logger.debug(f"Attempting to resolve the following unresolved df:\n{unresolved_df}")
    logger.debug(f"Using this mapping:\n{agq_expressions}")

    # Iterate through to solve the 'unresolved_df', if possible
    previous_unknowns = set()
    iteration = 0
    max_iterations = 100
    while iteration < max_iterations:
        unresolved_df["resolved_dependency"] = unresolved_df[
            "constituent_elements"
        ].apply(lambda x: resolve_elements_list(elements=x, status_df=status_df))

        status_df.loc[unresolved_df.index, STATUS_LABEL] = unresolved_df[
            "resolved_dependency"
        ]

        # Make note of current set of unknowns
        current_unknowns = set(
            unresolved_df[unresolved_df[STATUS_LABEL] == "unknown"].index
        )

        if not current_unknowns:
            logger.info("All dependencies resolved!")
            break
        elif current_unknowns == previous_unknowns:
            logger.warning(
                f"Unable to resolve the same set of unknowns: {current_unknowns} We have an error or circular dependency."
            )
            break

        previous_unknowns = current_unknowns
        unresolved_df = unresolved_df[unresolved_df["resolved_dependency"] == "unknown"]
        iteration += 1
    else:
        logger.warning(
            f"Reached maximum iterations ({max_iterations}) without resolving all dependencies."
        )

    return status_df


def set_model_parameters(param_set, *, model: bsc.COPASI.CDataModel):
    """
    Update model parameters and initial species concentrations based on the provided parameter set.

    The function checks each key in the parameter set and identifies if it's a model parameter or a species.
    It then updates the model accordingly.

    Parameters:
    - param_set (dict): Dictionary containing parameter names/species as keys and desired values/concentrations as values.
    """
    # Get current model parameters and species
    model_parameters = safe_df(bsc.get_parameters(model=model)).index.tolist()
    model_species = bsc.get_species(model=model).index.tolist()

    for param, value in param_set.items():
        if param in model_parameters:
            bsc.set_parameters(
                name=param, value=value, initial_value=value, model=model
            )
        elif param in model_species:
            bsc.set_species(
                name=param,
                initial_concentration=value,
                concentration=value,
                model=model,
            )


def initialise_status_df(*, model: bsc.COPASI.CDataModel):
    df_bsc_names = get_basico_outputs(model=model)
    # All variables depend on a 'core constituent' (themselves or something else)
    df_bsc_names["Core_Constituent"] = np.where(
        df_bsc_names["Transient"].notna(),
        df_bsc_names["Transient"],
        "(" + df_bsc_names.index + ")",
    )
    # We want each row to be a different possible variable
    status_df = df_bsc_names.melt(
        value_name="name",
        var_name="Descriptor",
        value_vars=["Initial", "Transient", "Rate", "Flux Concentration"],
        id_vars=["Core_Constituent", "Type"],
    )
    status_df.dropna(inplace=True)
    status_label = "independent/dependent/unknown"
    status_df[status_label] = None
    status_df.set_index("name", inplace=True)
    new_row = pd.DataFrame(
        {
            "Core_Constituent": "Time",
            "Type": "Time",
            "Descriptor": "Model Time",
            "independent/dependent/unknown": "independent",
        },
        index=["Time"],
    )
    status_df = pd.concat([status_df, new_row])
    return status_df


def extract_items_from_expression(expression):
    """
    Extracts items from a given mathematical expression.

    Parameters:
    - expression (str): The mathematical expression to extract items from.

    Returns:
    - list: A list of extracted items from the expression.

    Note:
    The function works by distinguishing operators and functions from 'items'.
    Items are constructed gradually and are only extracted once they have met two
    conditions. A, they must be followed by an operator/function. B, they must have an
    equal number of square brackets '[' and ']'.
    """
    # Define known operators and functions
    operators = [
        "+",
        "-",
        "/",
        "*",
        "%",
        "^",
        "and",
        "or",
        "xor",
        "not",
        "<=",
        "<",
        ">=",
        ">",
        "!=",
        "==",
        "abs",
        "floor",
        "ceil",
        "factorial",
        "log",
        "log10",
        "ABS",
        "FLOOR",
        "CEIL",
        "FACTORIAL",
        "LOG",
        "LOG10",
        "EXP",
    ]

    def remove_standalone_parentheses(expr: str):
        """Removes starting parentheses, ending parentheses and standalone parentheses surrounded by spaces."""

        return (
            expr.replace(" ( ", " ")
            .replace(" ) ", " ")
            .replace("( ", " ")
            .replace(" )", " ")
        )

    def has_balanced_brackets(string):
        """Checks if there's an equal number of '[' and ']' in a string."""
        return string.count("[") == string.count("]")

    logger.debug(f"Expression is: \n{expression}\n")
    expression = remove_standalone_parentheses(expression)
    logger.debug(f"Brackets are removed: \n{expression}\n")

    # Split the expression by space to begin the extraction process
    parts = expression.split()
    logger.debug(
        f"Expression is seperated into parts: \n{parts}\n"
        f"We now must find which parts are operators, items, or parts of an item\n"
    )

    extracted_items = []
    extracted_operators = []
    construction_list = []
    logger.debug(
        f"Lists to be used later: \n'operators':{operators}\n"
        f"'extracted_items':{extracted_items}\n"
        f"'extracted_operators':{extracted_operators}\n"
        f"'construction_list':{construction_list}\n"
    )

    # Iterate over the parts
    logger.debug("\nIterating through the parts:\n")
    for part in parts:
        construction_list_is_empty = bool(not construction_list)
        construction_is_balanced = has_balanced_brackets(" ".join(construction_list))

        # Non-operators are 'items'
        if part not in operators:
            construction_list.append(part)
            logger.debug(f"    '{part}' is appended to construction")
            continue

        # Is the operator part of an item?
        if construction_list_is_empty:
            extracted_operators.append(part)
            logger.debug(f"    '{part}' extracted as operator")
            continue

        # Is the fully constructed item balanced? part of an item?
        if construction_is_balanced:
            extracted_items.append(" ".join(construction_list))
            logger.debug(
                f"    construction, {construction_list}, is extracted as a complete item: {extracted_items}"
            )

            construction_list = []  # Reset the item parts
            logger.debug("    construction is reset to []")

            extracted_operators.append(
                part
            )  # Add the operator/function itself as an item
            logger.debug(f"    '{part}' extracted as operator")
        else:
            construction_list.append(part)
            logger.debug(f"    '{part}' is appended to construction")

    # Add any remaining parts as an item
    if construction_list:
        extracted_items.append(" ".join(construction_list))
        # logger.debug(f"construction, {construction_list}, is extracted as a complete item: {extracted_items}")

    # Remove any trailing or leading whitespace from the items
    extracted_items = [item.strip() for item in extracted_items]

    # Item could be a number
    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    extracted_items = [item for item in extracted_items if not is_number(item)]

    return extracted_items


def resolve_elements_list(elements, status_df):
    elements_present = {n: n in status_df.index for n in elements}
    if not all(elements_present.values()):
        raise ValueError(f"Not all elements in {elements_present.keys()} were present")

    dependencies = status_df.loc[elements, "independent/dependent/unknown"]

    if any(dependencies == "dependent"):
        result = "dependent"
    elif any(dependencies == "unknown"):
        result = "unknown"
    elif all(dependencies == "independent"):
        result = "independent"
    else:
        # Error handling
        logger.debug("Possibly encountered unresolvable set of dependencies")
        result = "error: elements not found in status dataframe"

    return result


def get_basico_outputs(
    display_parameters=True,
    display_species=True,
    display_reactions=True,
    display_compartments=True,
    *,
    model: bsc.COPASI.CDataModel | None = None,
):
    """
    Displays all/most elements that can be mapped to for time-course simulations or for parameter estimation etc.
    Set any argument to be True or False to see them or not in the output.
    You can specify the model if different to the loaded model.
    """
    df_list = []

    # Build Global Quantities Table
    if display_parameters:
        params = safe_df(bsc.get_parameters(model=model))
        if not params.empty:
            parameter_names = params.loc[:, ["display_name"]]
            parameter_names["Type"] = "Parameter"
            parameter_names["Initial"] = (
                parameter_names["display_name"] + ".InitialValue"
            )
            parameter_names["Transient"] = parameter_names["display_name"]
            parameter_names["Rate"] = parameter_names["display_name"] + ".Rate"
            parameter_names.drop(columns=["display_name"], inplace=True)
            df_list.append(parameter_names)

    # Build Species Concentrations Table
    if display_species:
        species = safe_df(bsc.get_species(model=model))
        if not species.empty:
            species_conc_names = species.loc[:, ["display_name"]]
            species_conc_names["Type"] = "Species Concentration"
            species_conc_names["Initial"] = (
                "[" + species_conc_names["display_name"] + "]_0"
            )
            species_conc_names["Transient"] = (
                "[" + species_conc_names["display_name"] + "]"
            )
            species_conc_names["Rate"] = species_conc_names["display_name"] + ".Rate"
            species_conc_names.drop(columns=["display_name"], inplace=True)
            species_conc_names.index = (
                "["
                + species_conc_names.index
                + "] ("
                + bsc.get_model_units()["quantity_unit"]
                + ")"
            )
            df_list.append(species_conc_names)

            # Build Species Particles Table
            species_particle_names = species.loc[:, ["display_name"]]
            species_particle_names["Type"] = "Species Particle"
            species_particle_names["Initial"] = (
                species_particle_names["display_name"] + ".InitialParticleNumber"
            )
            species_particle_names["Transient"] = species_particle_names["display_name"]
            species_particle_names["Rate"] = (
                species_particle_names["display_name"] + ".ParticleNumberRate"
            )
            species_particle_names.drop(columns=["display_name"], inplace=True)
            species_particle_names.index = (
                species_particle_names.index + " (Particle Number)"
            )
            df_list.append(species_particle_names)

    # Build Compartments Table
    if display_compartments:
        compartments = safe_df(bsc.get_compartments(model=model))
        if not compartments.empty:
            compartment_names = compartments.loc[:, ["display_name"]]
            compartment_names["Type"] = "Compartment"
            compartment_names["Initial"] = (
                compartment_names["display_name"] + ".InitialVolume"
            )
            compartment_names["Transient"] = compartment_names["display_name"]
            compartment_names["Rate"] = compartment_names["display_name"] + ".Rate"
            compartment_names.drop(columns=["display_name"], inplace=True)
            df_list.append(compartment_names)

    # Build Reactions Table
    if display_reactions:
        reactions = safe_df(bsc.get_reactions(model=model))
        if not reactions.empty:
            reactions_names = reactions.loc[:, ["display_name"]]
            reactions_names["Type"] = "Reaction"
            reactions_names["Flux Concentration"] = (
                reactions_names["display_name"] + ".Flux"
            )
            reactions_names["Flux Particle Number"] = (
                reactions_names["display_name"] + ".ParticleFlux"
            )
            reactions_names.drop(columns=["display_name"], inplace=True)
            df_list.append(reactions_names)

    return pd.concat(df_list) if df_list else pd.DataFrame()
