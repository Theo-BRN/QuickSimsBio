import hashlib
import itertools
import json
import logging
import pprint
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import basico as bsc
import numpy as np
import pandas as pd
import scipy.integrate

logger = logging.getLogger(__name__)
# Constants
DEFAULT_CACHE = Path.home() / ".mysimcache"
TIME_SECONDS_COLUMN = "Time (seconds)"
TIME_MINUTES_COLUMN = "Time (minutes)"
OUTPUT_VALUE_NAME = "Model_Output"
OUTPUT_TYPE_NAME = "Model_Output_Type"


def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Ensures a DataFrame is returned even if the source returns None."""
    return pd.DataFrame() if df is None else df


def create_parameter_scan_dict(
    model_path=None,
) -> dict[str, None | list[float | int] | dict[str, bool | float]]:
    """
    Create a dictionary with default 'no-change' values for all model parameters and species.

    Parameters:
    - model: The model object.

    Returns:
    - dict: A dictionary with parameters and species as keys and `None` as default values.
    """
    if model_path is not None:
        bsc.load_model(model_path)

    # Fetch all parameters and species from the model
    all_params = safe_df(bsc.get_parameters())
    if all_params.empty:
        all_fixed_params = pd.Series()
    else:
        all_fixed_params = all_params[all_params["type"] == "fixed"].index

    # Create the dictionary with default 'no-change' values
    fixed_param_dict = {param: None for param in all_fixed_params}
    species_dict = {species: None for species in safe_df(bsc.get_species()).index}

    # Merge the two dictionaries
    scan_dict = {**fixed_param_dict, **species_dict}

    return scan_dict


def generate_parameter_sets_df(scan_dict):
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


def set_model_parameters(param_set):
    """
    Update model parameters and initial species concentrations based on the provided parameter set.

    The function checks each key in the parameter set and identifies if it's a model parameter or a species.
    It then updates the model accordingly.

    Parameters:
    - param_set (dict): Dictionary containing parameter names/species as keys and desired values/concentrations as values.
    """
    # Get current model parameters and species
    model_parameters = safe_df(bsc.get_parameters()).index.tolist()
    model_species = bsc.get_species().index.tolist()

    for param, value in param_set.items():
        if param in model_parameters:
            bsc.set_parameters(name=param, value=value, initial_value=value)
        elif param in model_species:
            bsc.set_species(
                name=param, initial_concentration=value, concentration=value
            )


def run_individual_simulation_for_param_set(
    param_set, timepoints, output_selection, specified_FGQs=None, sim_num=None, **kwargs
):
    """
    This is a new version of run_simulation_for_param_set
    """
    set_model_parameters(param_set)

    # Run simulations with potential kwargs
    sim_df = bsc.run_time_course_with_output(
        values=timepoints,
        output_selection=output_selection,
        start_time=timepoints[0],
        **kwargs,
    )

    # Add simulation number
    if sim_num is not None:
        sim_df["Sim_Num"] = sim_num

    return sim_df


def run_simulations(
    scan_dict,
    timepoints,
    output_selection=None,
    model_path=None,
    specified_FGQs=None,
    format_output="wide",
    **kwargs,
):
    """
    This will eventually be an updated version of run_parameter_scan which uses run_time_course_with_output
    instead of run_time_course as it's based simulating function.

    This (will) automatically melt the dataframe and so additional independent or dependent variables are not naturally incorporated.
    Hence, they will need to be incorporated afterwards.
    """
    if model_path is not None:
        bsc.load_model(model_path)

    # Default output selection are the species
    if output_selection is None:
        species_names = bsc.get_species().index.tolist()
        output_selection = [f"[{species}]" for species in species_names]

    if "Time" not in output_selection:
        output_selection = ["Time"] + output_selection

    # Extract parameters that need to be varied from the scan_dict
    varying_params = {
        key: value for key, value in scan_dict.items() if value is not None
    }

    # scan_dict keys are in the format required to change with set_species() and set_parameter()
    param_sets_df = generate_parameter_sets_df(varying_params)

    # scan_dict keys should be in the output selection, but are formatted differently when extracted
    params = bsc.get_parameters()
    set_params = set([] if params is None else params.index)
    output_params = [
        f"Values[{key}]" for key in varying_params.keys() if key in set_params
    ]

    set_species = set(bsc.get_species().index)
    output_species = [
        f"[{key}]_0" for key in varying_params.keys() if key in set_species
    ]

    # (I think) the only things to change are the initial species concentrations or parameters, so will warn if anything else found
    output_neither_param_nor_species = [
        key for key in varying_params.keys() if key not in set_species | set_params
    ]
    if output_neither_param_nor_species:
        logger.warning(
            f"scan_dict items: {output_neither_param_nor_species} were not recognised as species or parameters"
        )

    output_selection = output_selection + output_params + output_species

    # Iterate over each parameter set
    sim_df_list = []
    for sim_num, (param_set_name, param_set) in enumerate(
        param_sets_df.items(), start=1
    ):
        sim_df = run_individual_simulation_for_param_set(
            param_set, timepoints, output_selection, specified_FGQs, sim_num, **kwargs
        )
        sim_df_list.append(sim_df)

    # Concatenate all individual simulation dataframes
    sim_df_all = pd.concat(sim_df_list, ignore_index=True)

    if format_output == "wide":
        return sim_df_all
    elif format_output != "long":
        print("Please supply either 'long' or 'wide' as the format output")
    else:
        return auto_melt(
            sim_df_all,
            additional_independent_vars=[
                "Sim_Num",
                TIME_SECONDS_COLUMN,
                TIME_MINUTES_COLUMN,
            ],
        )


class Congruence(Enum):
    UNKNOWN = auto()
    TRUE = auto()
    FALSE = auto()


class SimulationStatus(Enum):
    UNKNOWN = auto()
    NOT_RUN = auto()
    SUCCESS = auto()
    FAILED = auto()


class OutputCategory(Enum):
    # Parameter categories
    PARAM_INITIAL = "parameter initial value"
    PARAM_TRANSIENT = "parameter transient value"
    PARAM_RATE = "parameter rate"

    # Species (concentration) categories
    SPECIES_INITIAL = "species initial value"
    SPECIES_TRANSIENT = "species transient value"
    SPECIES_RATE = "species rate"

    # Reaction rate categories
    REACTION_RATE_CONC = "reaction rate (concentration)"
    REACTION_RATE_PARTICLE = "reaction rate (particle number)"

    # Species (particle number) categories
    SPECIES_PARTICLE_INITIAL = "species (particle number) initial value"
    SPECIES_PARTICLE_TRANSIENT = "species (particle number) transient value"
    SPECIES_PARTICLE_RATE = "species (particle number) rate"

    # Compartment categories
    COMPARTMENT_INITIAL = "compartment volume initial value"
    COMPARTMENT_TRANSIENT = "compartment volume transient value"
    COMPARTMENT_RATE = "compartment volume rate"

    # Time
    TIME = "Time (any)"


class ModelParameterSet:
    """
    Represents a parameter set combined with a particular model.
    Conceptually, this is a single point a single model's parameter space.
    """

    def __init__(
        self, model_path: str | None = None, parameters: dict | pd.Series | None = None
    ):
        self._model_path = None
        self.model_name = None
        self._parameters = None
        self._model_variables_dict = (
            None  # Model variables are either independent, dependent or unknown
        )
        self._timepoints = None
        self._simulated_timepoints = None
        self._output_selection = None
        self.output_categories = {cat.value: False for cat in OutputCategory}
        self.output_categories[OutputCategory.SPECIES_TRANSIENT.value] = True
        self.output_categories[OutputCategory.TIME.value] = True
        self.results = None
        self._results_long = None
        self._auc_results = None
        self._verified = False

        # Use setters for safety/validation
        if model_path is not None:
            self.model_path = model_path
        if parameters is not None:
            self.parameters = parameters

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, path: str):
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if path_obj.suffix.lower() != ".cps":
            raise ValueError(f"Expected a COPASI .cps file, got: {path_obj.suffix}")

        self._model_path = path

        if self.congruence is Congruence.TRUE:
            self._mark_verified()  # Model has become congruent, thus it is verified

        logger.info("Finished setting model_path")

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        was_verified = self.verified

        if isinstance(params, dict):
            params = pd.Series(params)
        elif not isinstance(params, pd.Series):
            raise TypeError("parameters must be a dict or pandas Series")

        # if model is already congruent, we can keep it congruent
        if self.model_congruence is Congruence.TRUE:
            params = self._expand_with_defaults(params)
            self._validate_parameters(params)
            self._parameters = params
        else:
            self._parameters = params
            logger.info(
                "Parameters stored as partial overrides (class instance not yet congruent)"
            )

        self._output_selection = (
            None  # new parameters should reset the output selection
        )

        if self.congruence is Congruence.TRUE:
            self._mark_verified()  # Model has become congruent, thus it is verified
        elif was_verified:
            self._invalidate_verification()  # Parameter update now break verification

    def _expand_with_defaults(self, overrides: pd.Series) -> pd.Series:
        """Return a full parameter set: defaults from basico + user overrides.
        TODO Consider replacing this function with it's potentially better
        alternative (follows)."""
        required = self.effective_inputs
        if required.empty:
            return overrides

        params_df = safe_df(bsc.get_parameters())
        species_df = safe_df(bsc.get_species())

        basico_default_parameter_values = params_df.loc[
            required.intersection(params_df.index), "initial_value"
        ]
        if basico_default_parameter_values.empty:
            raise ValueError("Cannot find default initial values for parameters")

        basico_default_species_values = bsc.get_species().loc[
            required.intersection(species_df.index), "initial_concentration"
        ]
        if basico_default_species_values.empty:
            raise ValueError("Cannot find default initial concentrations for species")

        basico_default_values = pd.concat(
            [basico_default_parameter_values, basico_default_species_values]
        )

        full = dict(basico_default_values)
        for name, value in overrides.items():
            if name not in full.keys():
                raise KeyError(
                    f"Unknown or irrelevant parameter: {name}, not found in {full.keys()}"
                )
            full[name] = value
        return pd.Series(full)

    # def _expand_with_defaults(self, overrides: pd.Series) -> pd.Series:
    #     """Return a full parameter set: defaults from basico + user overrides."""
    #     required = self.effective_inputs
    #     if required.empty:
    #         return overrides

    #     # 1. Fetch data once (avoid redundant bsc calls)
    #     params_df = safe_df(bsc.get_parameters())
    #     species_df = safe_df(bsc.get_species())

    #     # 2. Extract defaults using index alignment
    #     # We combine them first so we have a single 'master' defaults Series
    #     defaults = pd.concat(
    #         [params_df["initial_value"], species_df["initial_concentration"]]
    #     )

    #     # Filter defaults to only what is 'required'
    #     full_defaults = defaults.reindex(required)

    #     if full_defaults.isna().any():
    #         missing = full_defaults[full_defaults.isna()].index.tolist()
    #         raise ValueError(f"Could not find default values for: {missing}")

    #     # 3. The "Pandas Way" to merge: combine_first
    #     # This takes overrides, and fills in gaps with full_defaults.
    #     # It also naturally handles the case where overrides might contain keys
    #     # not in required (though we should validate that).

    #     extra_keys = overrides.index.difference(full_defaults.index)
    #     if not extra_keys.empty:
    #         raise KeyError(f"Unknown or irrelevant parameters: {extra_keys.tolist()}")

    #     return overrides.combine_first(full_defaults)

    def _parameters_aligned(self, rtol=1e-5, atol=0):
        """Check if all parameters in the series are present in bsc species or parameters with correct values."""

        basico_series = pd.concat(
            [
                safe_df(bsc.get_parameters()).loc[:, "initial_value"],
                safe_df(bsc.get_species()).loc[:, "initial_concentration"],
            ]
        )

        if self.parameters is None:
            raise ValueError(
                f"Tried to align parameters, but self.parameters is None: {self.parameters}"
            )

        for name, value in self.parameters.items():
            if name not in basico_series.index:
                logger.info(
                    f"parameter, {name}, was not found in the loaded basico model"
                )
                return False

            basico_value = basico_series.at[name]
            if not isinstance(basico_value, (int, float, np.number)):
                logger.warning(
                    f"Initial {name} is not a number in basico! Type: {type(basico_value)}"
                )
                return False

            # Added float(args) to narrow the types
            if not np.isclose(
                float(basico_value),
                float(value),
                rtol=float(rtol),
                atol=float(atol),
            ):
                logger.info(
                    f"the parameter value, {name}, was not sufficiently close to that in the loaded basico model: {basico_series.at[name]} vs {value}"
                )
                return False
        return True

    @property
    def parameter_congruence(self) -> Congruence:
        # TODO probably get rid of this property, and combine it into a Congruence that is not only true or false, but is maybe "alignment: (Nothing, Model, Model and Parameters)"
        logger.info("Checking model congruence to return parameter_congruence")
        m_congr = self.model_congruence
        if self.parameters is None:
            return Congruence.UNKNOWN
        elif m_congr is not Congruence.TRUE:
            return Congruence.UNKNOWN
        elif self._parameters_aligned():
            return Congruence.TRUE
        return Congruence.FALSE

    @property
    def model_congruence(self) -> Congruence:
        current_CDataModel = bsc.get_current_model()
        if current_CDataModel is None:
            logger.info("basico has no model loaded, so model is not congruent")
            return Congruence.FALSE

        if not self._model_path:
            logger.info("Returning model_congruence as UNKNOWN - no model_path set")
            return Congruence.UNKNOWN

        current_model_path = Path(current_CDataModel.getFileName()).resolve()
        stored_model_path = Path(self._model_path).resolve()
        if current_model_path != stored_model_path:
            logger.info(
                "Returning model_congruence as FALSE - model file not aligned to basico's loaded model file"
            )
            return Congruence.FALSE

        elif bsc.get_model_name() != self.model_name:
            # TODO we have a lot of calls here just from initiating a mps instance. This clogs up the logs a lot
            logger.info(
                f"Returning model_congruence as FALSE - basico's loaded model name '{bsc.get_model_name()}' is not the same as the Model Parameter Set's model name '{self.model_name}'"
            )
            return Congruence.FALSE
        else:
            return Congruence.TRUE

        logger.info("Unsure how we got here")
        return Congruence.UNKNOWN

    @property
    def congruence(self) -> Congruence:
        m_congr = self.model_congruence
        p_congr = self.parameter_congruence
        if m_congr is Congruence.TRUE and p_congr is Congruence.TRUE:
            return Congruence.TRUE
        elif m_congr is Congruence.FALSE or p_congr is Congruence.FALSE:
            return Congruence.FALSE
        return Congruence.UNKNOWN

    @property
    def verified(self) -> bool:
        return self._verified

    def _mark_verified(self):
        self._verified = True

    def _invalidate_verification(self):
        self._verified = False

    @property
    def simulation_status(self) -> SimulationStatus:
        if self.results is None:
            return SimulationStatus.NOT_RUN
        elif self._simulated_timepoints is None:
            logger.warning(
                "No simulated_timepoints were found, despite there being results"
            )
            return SimulationStatus.UNKNOWN
        elif all(pd.Series(self.timepoints).isin(self._simulated_timepoints)):
            return SimulationStatus.SUCCESS
        else:
            # The requested timepoints not being present in the data indicates simulation failure
            return SimulationStatus.FAILED

    def _validate_parameters(self, params: pd.Series):
        if self._parameters is not None and self.model_congruence is Congruence.TRUE:
            valid_names = safe_df(bsc.get_species()).index.union(
                safe_df(bsc.get_parameters()).query("type == 'fixed'").index
            )
            invalid = [name for name in params.index if name not in valid_names]
            if any(invalid):
                logger.warning(
                    "Invalid parameter names found in supplied params: {invalid}"
                )
            else:
                logger.info("Supplied parameters were all found to be valid")

    @staticmethod
    def _extract_reaction_params(df_reactions: pd.DataFrame) -> list:
        """
        Provides a list of
        """
        params = []
        for mapping in df_reactions["mapping"]:
            params.extend(mapping[k] for k in ["k1", "k2"] if k in mapping)
        print(f"_extract_reaction_params resulted with {params}")
        return params

    # TODO simplify this static method - why so many different types used?
    @staticmethod
    def _collect_expressions(df_params, df_events, reaction_params) -> list:
        reaction_assignment_expressions = df_params.loc[
            (
                df_params.index.isin(reaction_params)
                & (df_params["type"] == "assignment")
            ),
            "expression",
        ]

        reaction_params_fixed = df_params.loc[
            (df_params.index.isin(reaction_params) & (df_params["type"] == "fixed"))
        ].index

        event_expressions = []
        if df_events is not None:
            for assignments in df_events["assignments"]:
                for bit in assignments:
                    event_expressions.append(bit["expression"])

        expressions = [
            pd.concat(
                [
                    reaction_assignment_expressions,
                    pd.Series(reaction_params_fixed),
                    pd.Series(event_expressions),
                ]
            )
        ]
        return expressions

    @property
    def effective_inputs(self) -> pd.Index:
        """Return the subset of parameters/species that affect model outputs."""
        if self.model_congruence is not Congruence.TRUE:
            logger.info("Could not compute effective inputs without model congruence")
            return pd.Index([])

        df_params = safe_df(bsc.get_parameters())
        df_events = safe_df(bsc.get_events())
        df_species = safe_df(bsc.get_species())
        df_reactions = safe_df(bsc.get_reactions())

        reaction_params = self._extract_reaction_params(df_reactions)
        expressions = self._collect_expressions(
            df_params=df_params, df_events=df_events, reaction_params=reaction_params
        )

        expression_constituents = tuple(
            pd.Series(expressions).map(extract_items_from_expression)
        )
        inputs = set(np.hstack(expression_constituents).tolist())
        parameter_inputs = df_params.loc[
            df_params["display_name"].isin(inputs), :
        ].index

        species = df_species.index
        return pd.Index(parameter_inputs.union(species))

    @property
    def timepoints(self):
        return self._timepoints

    @timepoints.setter
    def timepoints(self, value):
        if value is None:
            self._timepoints = None
            return
        try:
            iterator = iter(value)
        except TypeError:
            raise TypeError("timepoints must be an iterable of numbers")
        try:
            processed = [float(x) for x in iterator]
        except (ValueError, TypeError):
            raise TypeError("timepoints must contain only ints or floats")
        self._timepoints = processed

    def display_events(self):
        """Print a summary of events in the model (name, trigger, assignments)."""
        if self.model_congruence is not Congruence.TRUE:
            raise ValueError(
                "State must be congruent for events to be correctly retrieved"
            )

        df_events = safe_df(bsc.get_events())
        if df_events.empty:
            print("No events defined in the current model.")
            return

        for name, row in df_events.iterrows():
            print(f"Event: {name}")
            print(f"  Trigger: {row['trigger']}")
            if row["assignments"]:
                print("  Assignments:")
                for assignment in row["assignments"]:
                    target = assignment.get("target")
                    expr = assignment.get("expression")
                    print(f"     {target} := {expr}")
            print()

    def _summarize_timepoints(self, max_items=10):
        """Return a string summary of timepoints (head ... tail)."""
        if self._timepoints is None:
            return "None"
        tp = self._timepoints
        n = len(tp)
        if n <= max_items:
            return pprint.pformat(tp)
        else:
            head = ", ".join(f"{x}" for x in tp[:5])
            tail = ", ".join(f"{x}" for x in tp[-3:])
            return f"[{head}, ..., {tail}] (n={n})"

    @property
    def output_selection(self):
        """
        Dynamically builds the list of output selections based on flags
        set in the self.output_categories dictionary.
        """
        if self._output_selection is not None:
            return self._output_selection

        selection = []
        if (
            self._output_selection is None
            and self.model_congruence is not Congruence.TRUE
        ):
            return selection

        if self._output_selection is None and self.model_congruence is Congruence.TRUE:
            species_names = safe_df(bsc.get_species()).index.tolist()
            parameter_names = safe_df(bsc.get_parameters()).index.tolist()
            reaction_names = safe_df(bsc.get_reactions()).index.tolist()
            compartment_names = safe_df(bsc.get_compartments()).index.tolist()

            if self.output_categories[OutputCategory.SPECIES_TRANSIENT.value]:
                selection.extend([f"[{s}]" for s in species_names])

            if self.output_categories[OutputCategory.SPECIES_INITIAL.value]:
                selection.extend([f"[{s}]_0" for s in species_names])

            if self.output_categories[OutputCategory.SPECIES_RATE.value]:
                selection.extend([f"{s}.Rate" for s in species_names])

            if self.output_categories[OutputCategory.SPECIES_PARTICLE_TRANSIENT.value]:
                selection.extend([f"{s}.ParticleNumber" for s in species_names])

            if self.output_categories[OutputCategory.SPECIES_PARTICLE_INITIAL.value]:
                selection.extend([f"{s}.InitialParticleNumber" for s in species_names])

            if self.output_categories[OutputCategory.SPECIES_PARTICLE_RATE.value]:
                selection.extend([f"{s}.ParticleNumberRate" for s in species_names])

            if self.output_categories[OutputCategory.PARAM_TRANSIENT.value]:
                selection.extend([f"Values[{p}]" for p in parameter_names])

            if self.output_categories[OutputCategory.PARAM_INITIAL.value]:
                selection.extend([f"Values[{p}].InitialValue" for p in parameter_names])

            if self.output_categories[OutputCategory.PARAM_RATE.value]:
                selection.extend([f"Values[{p}].Rate" for p in parameter_names])

            if self.output_categories[OutputCategory.REACTION_RATE_CONC.value]:
                # selection.extend([f"{{({r}).Flux}}" for r in reaction_names])
                selection.extend([f"({r}).Flux" for r in reaction_names])

            if self.output_categories[OutputCategory.REACTION_RATE_PARTICLE.value]:
                selection.extend([f"({r}).ParticleFlux" for r in reaction_names])
                # selection.extend([f"{{({r}).ParticleFlux}}" for r in reaction_names])

            if self.output_categories[OutputCategory.COMPARTMENT_TRANSIENT.value]:
                selection.extend([f"Compartments[{c}]" for c in compartment_names])

            if self.output_categories[OutputCategory.COMPARTMENT_INITIAL.value]:
                selection.extend(
                    [f"Compartments[{c}].InitialVolume" for c in compartment_names]
                )

            if self.output_categories[OutputCategory.COMPARTMENT_RATE.value]:
                selection.extend([f"Compartments[{c}].Rate" for c in compartment_names])

            if self.output_categories[OutputCategory.TIME.value]:
                for each_time_var in [TIME_MINUTES_COLUMN, TIME_SECONDS_COLUMN]:
                    full_name = f"Values[{each_time_var}]"
                    if (
                        each_time_var in parameter_names and full_name not in selection
                    ):  # May already be present if PARAM_TRANSIENT is True
                        selection.append(full_name)
                if not any(
                    f"Values[{time_var}]" in selection
                    for time_var in [TIME_MINUTES_COLUMN, TIME_SECONDS_COLUMN]
                ):
                    selection.append("Time")

            # The self.parameters should be in the selection (but don't need to be added twice)
            if self._parameters is None:
                raise ValueError(
                    f"Could not assess output selection as parameters were not defined. self._parameters: {self._parameters}"
                )
            changed_params = [
                f"Values[{param}].InitialValue"
                for param in self._parameters.keys()
                if param in parameter_names
            ]
            changed_initial_species = [
                f"[{species}]_0"
                for species in self._parameters.keys()
                if species in species_names
            ]
            model_params = [
                p
                for p in changed_params + changed_initial_species
                if p not in selection
            ]
            selection.extend(model_params)

            self._output_selection = selection

        return self._output_selection

    @output_selection.setter
    def output_selection(self, values):
        if values is None:
            self._output_selection = None
        else:
            if not (
                isinstance(values, list)
                or all(isinstance(each_value, str) for each_value in values)
            ):
                raise TypeError("output_selection must be a list of strings")
            logger.info(
                f"The output selection was updated from {self._output_selection} to {list(values)}"
            )
            self._output_selection = list(values)

    def _summarize_output_selection(self, max_items=10):
        sel = self._output_selection
        if sel is None:
            return "None"
        n = len(sel)
        if n <= max_items:
            return pprint.pformat(sel)
        else:
            head = ", ".join(sel[:5])
            tail = ", ".join(sel[-3:])
            return f"[{head}, ..., {tail}] (n={n})"

    def align_basico(self):
        """
        Will make ModelParameterSet instance congruent with basico model if possible.
        """
        if self.congruence is Congruence.TRUE:
            logger.warning("basico model already aligned with class instance")

        else:
            if self.model_congruence is not Congruence.TRUE:
                self._load_model()
            if self.parameter_congruence is not Congruence.TRUE:
                if self._parameters is None:
                    logger.warning(
                        "No parameters provided before alignment, so using model defaults"
                    )
                    self._parameters = pd.Series({})
                self._parameters = self._expand_with_defaults(self._parameters)
                self._validate_parameters(self._parameters)
                set_model_parameters(self._parameters)

            if self.congruence is Congruence.TRUE:
                self._mark_verified()  # Model has become congruent, thus it is verified
            else:
                logger.warning(
                    "align_basico is not working: did not make the model congruent"
                )
                if self.model_congruence is Congruence.FALSE:
                    logger.warning("The model was not made congruent")
                if self.parameter_congruence is Congruence.FALSE:
                    logger.warning(
                        f"The parameters were not made congruent: parameter_congruence is {self.parameter_congruence}"
                    )

    def _load_model(self):
        if not self._model_path:
            raise ValueError("No model_path set")
        bsc.load_model(self._model_path)
        self.model_name = bsc.get_model_name()
        self._model_variables_dict = get_dependency_dictionary()

    def _update_timepoints(
        self, start: None | float = None, step=None, stop=None, timepoints=None
    ):
        # If timepoints are supplied here, they overwrite the class' timepoints
        arange_inputs = {"start": start, "step": step, "stop": stop}
        if timepoints is not None:
            self.timepoints = timepoints
            logger.info("Set new timepoints")

        elif (start is not None) and (step is not None) and (stop is not None):
            self.timepoints = np.arange(start, stop, step)
            logger.info("Calculated new timepoints from provided start, step and stop.")

        elif self.timepoints is not None:
            logger.info("using stored timepoints for simulation")

        elif self.timepoints is None:
            print("⚠️ No timepoints set — defaulting to 0 → 1000 s (n=300)")
            self.timepoints = np.linspace(0, 1000, 300).tolist()

        # edge case handling
        elif any(arange_inputs.values()) and not all(arange_inputs.values()):
            remaining = [s for s, v in arange_inputs.items() if v is None]
            raise ValueError(
                f"Please supply remaining np.arange vars {' and '.join(remaining)}"
            )

    def run_sim(
        self, start=None, step=None, stop=None, timepoints=None, **solver_kwargs
    ):
        """
        Run a simulation with optional solver hyperparameters.
        Unlike `run_or_load`, this always forces a fresh run.

        Parameters
        ----------
        start, step, stop : float
            Time range definition (if `timepoints` not provided).
        timepoints : list[float], optional
            Explicit timepoints to simulate at.
        prettify_names : bool, default=True
            Whether to apply nicer column names to results.
        solver_kwargs : dict
            Passed directly to basico.run_time_course_with_output,
            e.g. method="radau5", a_tol=1e-9, r_tol=1e-5, max_steps=100000.
        """

        if self.congruence is not Congruence.TRUE:
            raise ValueError(
                f"{self} not aligned with basico before simulation request"
            )

        self._update_timepoints(
            start=start, step=step, stop=stop, timepoints=timepoints
        )
        if self._timepoints is None:
            raise ValueError("Timepoints not updated when simulation run (still None).")

        # ensure output selection is built
        _ = self.output_selection
        if self._output_selection is None:
            raise ValueError("output selection not defined, simulation not run")

        # It's useful to store the intended timepoints, to check if simulation is successful
        if "Time" not in self._output_selection:
            sim_selection = self._output_selection + ["Time"]
        else:
            sim_selection = self._output_selection

        if solver_kwargs:
            logger.info(f"Running simulation with solver kwargs: {solver_kwargs}")

        sim_df = bsc.run_time_course_with_output(
            values=self._timepoints,
            output_selection=sim_selection,
            start_time=self._timepoints[0],
            **solver_kwargs,
        )

        self._simulated_timepoints = sim_df["Time"].tolist()
        if "Time" not in self._output_selection:
            sim_df.drop(columns=["Time"], inplace=True)

        self.results = sim_df
        return self.results

    def inputs_dict(self) -> dict:
        """Return the dictionary of inputs that define the simulation (used for hashing)."""
        return {
            "model_path": str(self._model_path) if self._model_path else None,
            "parameters": self._parameters if self._parameters is not None else None,
            "timepoints": self._timepoints,
            "output_categories": self.output_categories,
        }

    def metadata_dict(self) -> dict:
        """Return a dictionary of metadata about the simulation run (not hashed)."""
        return {
            "simulated_timepoints": self._simulated_timepoints,
            # future metadata: runtime, solver, status, etc.
        }

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of metadata (no results)."""
        return {
            "model_path": str(self._model_path) if self._model_path else None,
            "parameters": self._parameters if self._parameters is not None else None,
            "timepoints": self._timepoints,
            "simulated_timepoints": self._simulated_timepoints,
            "output_categories": self.output_categories,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Rebuild a ModelParameterSet from metadata (results not included)."""
        return cls(
            model_path=data.get("model_path"),
            parameters=data.get("parameters"),
        )

    def _make_hash(self) -> str:
        """Generate a stable hash of the inputs for caching."""
        meta_str = json.dumps(self.inputs_dict(), sort_keys=True)
        return hashlib.sha1(meta_str.encode()).hexdigest()

    def save(self, cache_dir=None):
        """Save metadata (JSON) + results (Parquet) to cache."""
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        cache_dir.mkdir(parents=True, exist_ok=True)
        h = self._make_hash()

        combined_dict = {"inputs": self.inputs_dict(), "metadata": self.metadata_dict()}

        meta_path = cache_dir / f"{h}.json"
        with open(meta_path, "w") as f:
            json.dump(combined_dict, f, indent=2)

        # Save results only if successful
        if (
            self.results is not None
            and self.simulation_status is SimulationStatus.SUCCESS
        ):
            results_path = cache_dir / f"{h}.parquet"
            self.results.to_parquet(results_path, index=False)

    @classmethod
    def load(cls, h: str, cache_dir=None):
        """Load metadata + results by hash from cache."""
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        meta_path = cache_dir / f"{h}.json"
        results_path = cache_dir / f"{h}.parquet"

        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata found for hash {h}")

        with open(meta_path) as f:
            combined = json.load(f)

        instance = cls.from_dict(combined["inputs"])
        metadata = combined.get("metadata", {})
        instance._simulated_timepoints = metadata.get("simulated_timepoints", None)

        if results_path.exists():
            instance.results = pd.read_parquet(results_path)

        return instance

    def run_or_load(
        self, start=None, step=None, stop=None, timepoints=None, cache_dir=None
    ):
        """
        Run the simulation or load cached results if available.
        """
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._update_timepoints(
            start=start, step=step, stop=stop, timepoints=timepoints
        )
        h = self._make_hash()

        meta_path = cache_dir / f"{h}.json"
        results_path = cache_dir / f"{h}.parquet"

        if meta_path.exists() and results_path.exists():
            logger.info(f"[CACHE HIT] Loading cached results for hash {h}")

            with open(meta_path) as f:
                combined = json.load(f)
            metadata = combined.get("metadata", {})

            self._simulated_timepoints = metadata.get("simulated_timepoints", None)
            self.results = pd.read_parquet(results_path)

            return self.results
        else:
            # Run + save
            logger.info(f"[CACHE MISS] Running simulation for hash {h}")
            sim_df = self.run_sim()
            self.save(cache_dir=cache_dir)
            return sim_df

    @property
    def results_wide(self):
        """Return the raw wide-format simulation results."""
        return self.results

    @property
    def results_long(self):
        """Return the melted long-format simulation results."""
        if self._results_long is None:
            if (
                self._model_variables_dict is None
                and self.model_congruence is not Congruence.TRUE
            ):
                raise ValueError(
                    "Cannot automatically melt results dataframe as the class instance does not know the independent and dependent variables"
                )
            self._results_long = auto_melt(
                self.results, vars_dict=self._model_variables_dict
            )
        return self._results_long

    def results_baseline_corrected(
        self,
        mode="subtract",
        time_col=f"Values[{TIME_SECONDS_COLUMN}]",
        baseline_timepoint=0.0,
    ):
        """
        Return baseline-corrected long-format results.

        Parameters
        ----------
        mode : {"subtract", "divide"}
            How to correct the baseline:
            - "subtract": subtract baseline mean
            - "divide": divide by baseline mean (normalisation)
        baseline_timepoint : float
            Timepoint (in seconds) to use for baseline estimation.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with corrected values (column: "Corrected Value").
        """
        df = self.results_long.copy()

        corrected = []
        for output, group in df.groupby(OUTPUT_TYPE_NAME):
            baseline_val = group.loc[
                group[time_col] == baseline_timepoint, OUTPUT_VALUE_NAME
            ]
            if baseline_val.empty:
                raise ValueError(
                    f"No timepoints found in baseline_timepoint={baseline_timepoint} for {output}"
                )

            baseline = baseline_val.values[0]

            group = group.copy()
            if mode == "subtract":
                group[f"∆{OUTPUT_VALUE_NAME}"] = group[OUTPUT_VALUE_NAME] - baseline
            elif mode == "divide":
                group[f"Fold_{OUTPUT_VALUE_NAME}"] = group[OUTPUT_VALUE_NAME] / baseline
            else:
                raise ValueError(f"Unknown baseline correction mode: {mode}")
            corrected.append(group)

        return pd.concat(corrected, ignore_index=True)

    def auc_results(
        self,
        baseline_mode: str | None = None,
        baseline_timepoint: float | None = None,
        time_col: str = f"Values[{TIME_SECONDS_COLUMN}]",
    ):
        """
        Compute AUC for each output in the simulation.

        Parameters
        ----------
        baseline_mode : {None, "subtract", "divide"}, optional
            If None, AUC is computed directly on raw values.
            If "subtract" or "divide", values are baseline-corrected
            relative to `baseline_timepoint`.
        baseline_timepoint : float, optional
            Timepoint (in seconds) used for baseline correction.
            Required if baseline_mode is not None.
        time_col : str, default="Values[Time (s)]"
            Column representing time, used for baseline correction and for auc calculation

        Returns
        -------
        pd.DataFrame
            AUC summary in long format, one row per output.
        """
        # Choose which results to use
        if baseline_mode is None:
            df = self.results_long.copy()
            value_col = OUTPUT_VALUE_NAME
        else:
            if baseline_timepoint is None:
                raise ValueError(
                    "baseline_timepoint must be provided if baseline_mode is used."
                )
            df = self.results_baseline_corrected(
                mode=baseline_mode,
                time_col=time_col,
                baseline_timepoint=baseline_timepoint,
            )
            if baseline_mode == "subtract":
                value_col = f"∆{OUTPUT_VALUE_NAME}"
            elif baseline_mode == "divide":
                value_col = f"Fold_{OUTPUT_VALUE_NAME}"
            else:
                raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

        # Independent variables (other than time) shouldn't vary
        dep_vars_and_time = [
            f"Values[{TIME_MINUTES_COLUMN}]",
            f"Values[{TIME_SECONDS_COLUMN}]",
            OUTPUT_TYPE_NAME,
            OUTPUT_VALUE_NAME,
            value_col,
        ]

        df_no_output_cols = df.drop(
            columns=dep_vars_and_time, errors="ignore"
        ).drop_duplicates()
        if df_no_output_cols.shape[0] != 1:
            raise ValueError(
                f"Independent variables vary across — please check {OUTPUT_VALUE_NAME}"
            )

        calculated_auc_values = []
        for each_name, group in df.groupby(OUTPUT_TYPE_NAME):
            x = group[time_col]
            y = group[value_col]
            new_df = df_no_output_cols.copy()
            new_df[OUTPUT_TYPE_NAME] = each_name
            if baseline_mode is None:
                auc_col_name = "AUC"
            else:
                auc_col_name = f"AUC_{baseline_mode}_from_{baseline_timepoint}"
            new_df[auc_col_name] = scipy.integrate.trapezoid(y, x)
            new_df["n_timepoints"] = len(x)
            calculated_auc_values.append(new_df)

        return pd.concat(calculated_auc_values).reset_index(drop=True)

    def __repr__(self) -> str:
        # Model info
        if self._model_path is not None:
            model_status = self.model_name or "(Path provided)"
        else:
            model_status = "None set"

        # Parameter summary
        param_status = (
            f"{len(self._parameters) if self._parameters is not None else 0} set"
        )

        # Timepoints + outputs
        timepoint_summary = self._summarize_timepoints()
        output_summary = self._summarize_output_selection()

        # Build the string
        return (
            f"<ModelParameterSet "
            f"Model={model_status!r}, "
            f"Parameters={param_status}, "
            f"Verified={self.verified}, "
            f"Congruence={self.congruence.name}, "
            f"Simulation={self.simulation_status.name}>\n"
            f"  Timepoints: {timepoint_summary}\n"
            f"  Output selection: {output_summary}\n"
        )


def auto_melt(
    simulation_df,
    vars_dict=None,
    var_name=OUTPUT_TYPE_NAME,
    value_name=OUTPUT_VALUE_NAME,
):

    if vars_dict is None:
        vars_dict = get_dependency_dictionary()
        logger.debug(
            f"generating variables dictionary based on currently loaded model: '{bsc.get_model_name()}'"
        )

    independent_vars, dependent_vars, remaining_unknown_vars = vars_dict.values()

    if len(remaining_unknown_vars) > 0:
        logger.debug(
            f"some variable names did not have a resolved dependency status: {remaining_unknown_vars}"
        )

    logger.debug(
        f"auto_melt function is trying to melt the simulation_df, with columns\n {simulation_df.columns}"
    )
    logger.debug(
        f"Using the independent_vars: {independent_vars}\nthe dependent_vars: {dependent_vars}"
    )

    # pd.melt breaks if there are duplicate indepdenent_vars columns!
    duplicate_col_names = simulation_df.columns[
        simulation_df.columns.duplicated()
    ].unique()
    if len(duplicate_col_names) > 0:
        for col in set(duplicate_col_names):
            duplicates = simulation_df.loc[:, simulation_df.columns == col]
            if (duplicates.nunique(axis=1) == 1).all():
                logger.debug(
                    f"The column {col} appears more than once, but with identical data, duplicates will be removed."
                )
            else:
                raise ValueError(
                    f"The column {col} appears more than once, containing different data, unsure which columns to remove."
                )
        # reset df to not include duplicate cols
        simulation_df = simulation_df.loc[:, ~simulation_df.columns.duplicated()].copy()

    # pd.melt breaks if id_vars have duplicate (but not if there are duplicate value_vars)
    independent_vars = list(
        dict.fromkeys(independent_vars)
    )  # preserves order, removes duplicates

    # pd.melt breaks if the supplied id_vars or value_vars are not present in the df
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


def get_dependency_dictionary():
    status_df = get_dependency_status_dataframe()
    status_label = "independent/dependent/unknown"
    dependency_labels = ["independent", "dependent", "unknown"]
    dep_dict = {}
    for each_label in dependency_labels:
        dep_dict[each_label] = status_df.index[status_df[status_label] == each_label]

    return dep_dict


def get_dependency_status_dataframe():
    status_df = create_status_df()
    status_df = resolve_status_df(status_df)

    return status_df


def resolve_status_df(df):
    """
    Function takes in a status_df made by `create_status_df()` and resolves any unknown dependencies.
    These unknown dependencies arise from assigned parameters or compartments.
    """
    param_df = safe_df(bsc.get_parameters())
    param_df.set_index("display_name", inplace=True)
    agq_df = param_df[param_df["type"] == "assignment"].to_dict()
    agq_expressions = agq_df["expression"]

    status_label = "independent/dependent/unknown"
    unresolved_df = df.loc[df[status_label] == "unknown", :].copy()
    unresolved_df["expression"] = unresolved_df["Core_Constituent"].map(
        agq_expressions
    )  # adds expression/formula for variable
    unresolved_df["constituent_elements"] = unresolved_df["expression"].map(
        extract_items_from_expression
    )  # splits expression into list of elements

    logger.debug(f"Attempting to resolve the following unresolved df:\n{unresolved_df}")
    logger.debug(f"Using this mapping:\n{agq_expressions}")

    # Iterate through to solve the 'unresolved_df', if possible
    previous_unknowns = set()
    iteration = 0
    max_iterations = 100
    while iteration < max_iterations:
        unresolved_df["resolved_dependency"] = unresolved_df[
            "constituent_elements"
        ].apply(lambda x: resolve_elements_list(elements=x, status_df=df))

        df.loc[unresolved_df.index, status_label] = unresolved_df["resolved_dependency"]

        # Make note of current set of unknowns
        current_unknowns = set(
            unresolved_df[unresolved_df[status_label] == "unknown"].index
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

    return df


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


# def extract_items_from_expression(expression):
#     """
#     Extracts variable/species names using SymPy.
#     """
#     logger.info(f"temp: provided expression: {expression}")

#     if not isinstance(expression, str):
#         if isinstance(expression, pd.Series):
#             logger.warning("Expression provided as Series, extracting first value.")
#             expression = "" if expression.empty else expression.iloc[0]
#         else:
#             return []

#     try:
#         # 1. Parse the expression into a SymPy object
#         # evaluate=False prevents SymPy from trying to simplify it immediately
#         expr_obj = parse_expr(expression, evaluate=False)

#         # 2. free_symbols returns everything that is a variable (not a number/operator)
#         # We convert them to strings and return as a list
#         return [str(s) for s in expr_obj.free_symbols]

#     except Exception as e:
#         logger.error(f"SymPy could not parse expression: {expression}. Error: {e}")
#         return []


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


def create_status_df():
    status_df = initialise_status_df()
    status_label = "independent/dependent/unknown"
    status_df.loc[
        (status_df["Type"] == "Species Concentration")
        & (status_df["Descriptor"] == "Initial"),
        status_label,
    ] = "independent"
    status_df.loc[
        (status_df["Type"] == "Species Concentration")
        & (status_df["Descriptor"] != "Initial"),
        status_label,
    ] = "dependent"
    status_df.loc[
        (status_df["Type"] == "Species Particle")
        & (status_df["Descriptor"] == "Initial"),
        status_label,
    ] = "independent"
    status_df.loc[
        (status_df["Type"] == "Species Particle")
        & (status_df["Descriptor"] != "Initial"),
        status_label,
    ] = "dependent"
    status_df.loc[status_df["Type"] == "Reaction", status_label] = "dependent"

    # Parameters can be fixed or assigned
    params_or_none = bsc.get_parameters()
    all_params_df = params_or_none if params_or_none is not None else pd.DataFrame()
    fixed_params = all_params_df.loc[all_params_df["type"] == "fixed", "display_name"]
    is_fixed_param = (status_df["Type"] == "Parameter") & (
        status_df["Core_Constituent"].isin(fixed_params)
    )
    status_df.loc[is_fixed_param, ["Type"]] = "Fixed Parameter"
    status_df.loc[is_fixed_param, status_label] = "independent"
    assigned_params = all_params_df.loc[
        all_params_df["type"] == "assignment", "display_name"
    ]
    is_assigned_param = (status_df["Type"] == "Parameter") & (
        status_df["Core_Constituent"].isin(assigned_params)
    )
    status_df.loc[is_assigned_param, ["Type"]] = "Assigned Parameter"
    status_df.loc[is_assigned_param, status_label] = "unknown"

    # The Compartment Volume can be fixed or assigned
    all_comps = safe_df(bsc.get_compartments())
    fixed_comps = all_comps.loc[all_comps["type"] == "fixed", "display_name"]
    is_fixed_comp = (status_df["Type"] == "Compartment") & (
        status_df["Core_Constituent"].isin(fixed_comps)
    )
    status_df.loc[is_fixed_comp, ["Type"]] = "Fixed Compartment"
    status_df.loc[is_fixed_comp, status_label] = "independent"
    assigned_comps = all_comps.loc[all_comps["type"] == "assignment", "display_name"]
    is_assigned_comp = (status_df["Type"] == "Compartment") & (
        status_df["Core_Constituent"].isin(assigned_comps)
    )
    status_df.loc[is_assigned_comp, ["Type"]] = "Assigned Compartment"
    status_df.loc[is_assigned_comp, status_label] = "unknown"

    return status_df


def initialise_status_df():
    df_bsc_names = get_basico_names()
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


def get_basico_names(
    display_parameters=True,
    display_species=True,
    display_reactions=True,
    display_compartments=True,
    model=None,
):
    """
    Displays all/most elements that can be mapped to for time-course simulations or for parameter estimation etc.
    Set any argument to be True or False to see them or not in the output.
    You can specify the model if different to the loaded model.
    """
    # Load separate model if supplied
    if model is not None:
        bsc.load_model(model)

    # flag_list = [display_parameters, display_species, display_compartments, display_reactions]
    df_list = []

    # Build Global Quantities Table
    if display_parameters:
        parameter_names = safe_df(bsc.get_parameters()).loc[:, ["display_name"]]
        parameter_names["Type"] = "Parameter"
        parameter_names["Initial"] = parameter_names["display_name"] + ".InitialValue"
        parameter_names["Transient"] = parameter_names["display_name"]
        parameter_names["Rate"] = parameter_names["display_name"] + ".Rate"
        parameter_names.drop(columns=["display_name"], inplace=True)
        df_list.append(parameter_names)

    # Build Species Concentrations Table
    if display_species:
        species_conc_names = bsc.get_species().loc[:, ["display_name"]]
        species_conc_names["Type"] = "Species Concentration"
        species_conc_names["Initial"] = "[" + species_conc_names["display_name"] + "]_0"
        species_conc_names["Transient"] = "[" + species_conc_names["display_name"] + "]"
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
        species_particle_names = bsc.get_species().loc[:, ["display_name"]]
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
        compartment_names = safe_df(bsc.get_compartments()).loc[:, ["display_name"]]
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
        reactions_names = safe_df(bsc.get_reactions()).loc[:, ["display_name"]]
        reactions_names["Type"] = "Reaction"
        reactions_names["Flux Concentration"] = (
            reactions_names["display_name"] + ".Flux"
        )
        reactions_names["Flux Particle Number"] = (
            reactions_names["display_name"] + ".ParticleFlux"
        )
        reactions_names.drop(columns=["display_name"], inplace=True)
        df_list.append(reactions_names)

    return pd.concat(df_list)
