import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import qmc

from quicksimsbio.simulations import generate_parameter_sets_df

logger = logging.getLogger(__name__)


def generate_sobol_random(df_bounds, n_samples):
    sampler = qmc.Sobol(d=len(df_bounds.index), scramble=True, seed=3)
    sample = sampler.random(n=n_samples)
    scaled_sample = qmc.scale(sample, df_bounds["lower"], df_bounds["upper"])
    return pd.DataFrame(scaled_sample, columns=df_bounds.index)


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


def generate_input_space(scan_dict, n_rand_samples, print_summary=None):
    cats_dict = categorise_scan_dict(scan_dict)

    rand_dict = cats_dict["random_params"]

    def log_dict(d):
        new_dict = {}
        new_dict["upper"] = np.log10(d["upper"])
        new_dict["lower"] = np.log10(d["lower"])
        return new_dict

    logged_params = []
    sampling_dict = {}

    for param_key, param_value in rand_dict.items():
        if param_value["log"]:
            sampling_dict[f"log_{param_key}"] = log_dict(param_value)
            logged_params.append(param_key)
        else:
            sampling_dict[param_key] = {
                "upper": param_value["upper"],
                "lower": param_value["lower"],
            }

    df_bounds = pd.DataFrame(sampling_dict).transpose()
    sobol_samples = generate_sobol_random(df_bounds, n_rand_samples)

    cols_to_drop = []
    for each_col in sobol_samples:
        if each_col.startswith("log_") and each_col[4:] in logged_params:
            sobol_samples[each_col[4:]] = 10.0 ** sobol_samples[each_col]
            cols_to_drop.append(each_col)

    sobol_samples.drop(columns=cols_to_drop, inplace=True)
    grid_samples = generate_parameter_sets_df(
        {**cats_dict["individual_params"], **cats_dict["ranged_params"]}
    )

    return combine_param_sets(grid_samples, sobol_samples.transpose())
