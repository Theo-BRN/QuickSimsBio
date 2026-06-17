import logging

import numpy as np
import pandas as pd
import scipy.integrate
from scipy.optimize import curve_fit

TIME_SECONDS_COLUMN = "Time (seconds)"
TIME_MINUTES_COLUMN = "Time (minutes)"
OUTPUT_VALUE_NAME = "Model_Output"
OUTPUT_TYPE_NAME = "Model_Output_Type"
logger = logging.getLogger(__name__)


def straight_line(x, m, c):
    """
    A straight line function.

    Parameters
    ----------
    x : float or array-like
        The independent variable.
    m : float
        The gradient.
    c : float
        The y-intercept

    Returns
    -------
    float or array-like
        The value of the function m * x + c.
    """
    return m * x + c


def logistic_3_parameter_not_logged(x, Top, Bottom, EC50):
    """The simple Dose-response curve formulation of the 3-parameter
    logistic equation, with x given as the logged concentration."""
    return (Bottom) + (x * (Top - Bottom) / (EC50 + x))


def logistic_4_parameter_not_logged(x, Top, Bottom, EC50, Hillslope):
    """The simple Dose-response curve formulation of the 3-parameter
    logistic equation, with x given as the logged concentration."""
    return (Bottom) + (
        (x**Hillslope) * (Top - Bottom) / ((EC50**Hillslope) + (x**Hillslope))
    )


def operational_model(x, Emax, K_A, tau, n=1):
    """The Operational Model of Pharmacological Agonism
    with x given as the concentration."""
    return (Emax * (x**n) * (tau**n)) / ((K_A + x) ** n + (tau * (x**n)))


def rectangular_hyperbola(x, max, k):
    """
    General rectangular-hyperbola saturation function.

    Parameters
    ----------
    x : float or array-like
        The independent variable.
    max : float
        The asymptotic maximum value (analogous to Bmax or Vmax).
    k : float
        The half-saturation constant (value of x where the function
        reaches half of max). Analogous to Kd, Km, etc.

    Returns
    -------
    float or array-like
        The value of the function max * x / (k + x).
    """
    return max * x / (k + x)


def calc_initial_logistic_guess(x_values, y_values):
    """
    Estimates initial parameter guesses for logistic curve fitting.

    Args:
        x_values (array-like): Independent variable values.
        y_values (array-like): Dependent variable values.

    Returns:
        list: Initial guesses for logistic model parameters.
    """
    y_max, y_min = max(y_values), min(y_values)
    y_mid = (y_max + y_min) / 2
    closest_index = np.argmin(np.abs(y_values - y_mid))
    y_at_mid_x = x_values.iloc[closest_index]

    return [y_max, y_min, y_at_mid_x]


def calc_initial_rectangular_hyperbola_guess(x_values, y_values):
    """
    Simple initial guesses for rectangular hyperbola parameters.

    Parameters
    ----------
    Args:
        x_values (array-like): Independent variable values.
        y_values (array-like): Dependent variable values.

    Returns
    -------
    list
        Initial guesses [max_guess, k_guess].
    """
    x = np.asarray(x_values)
    y = np.asarray(y_values)

    # Guess for max (asymptote)
    max_guess = np.max(y)

    # Half-max value
    half_max = max_guess / 2.0

    # Find x closest to half-max
    idx = np.argmin(np.abs(y - half_max))
    k_guess = x[idx]

    # Fallback if data never gets near half-max
    if not np.isfinite(k_guess) or k_guess <= 0:
        k_guess = np.median(x)

    return [max_guess, k_guess]


def calc_logistic_bounds(x_values, y_values):
    """
    Estimates reasonable bounds for logistic curve fitting.

    Args:
        x_values (array-like): Independent variable values.
        y_values (array-like): Dependent variable values.

    Returns:
        tuple: (lower_bounds, upper_bounds) for logistic model parameters.
    """

    y_max = float(np.max(y_values))
    y_min = float(np.min(y_values))

    emax_upper = y_max * 10.0 if y_max != 0 else 1.0
    emin_upper = y_max * 10.0 if y_max != 0 else 1.0
    emax_lower = y_max * 0.1 if y_min != 0 else -1.0
    emin_lower = y_min * 0.1 if y_min != 0 else -1.0

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    ec50_lower = x_min
    ec50_upper = x_max

    # If later used for 4PL, hillslope bounds will be added elsewhere.
    lower_bounds = [emax_lower, emin_lower, ec50_lower]
    upper_bounds = [emax_upper, emin_upper, ec50_upper]

    return (lower_bounds, upper_bounds)


def calc_initial_straight_line_guess(x_values, y_values):
    """
    Simple initial guesses for straight line parameters.

    Parameters
    ----------
    Args:
        x_values (array-like): Independent variable values.
        y_values (array-like): Dependent variable values.

    Returns
    -------
    list
        Initial guesses [m_guess, c_guess].
    """
    max_x_point = (np.max(x_values), y_values.iloc[np.argmax(x_values)])
    min_x_point = (np.min(x_values), y_values.iloc[np.argmin(x_values)])
    max_y_point = (x_values.iloc[np.argmax(y_values)], np.max(y_values))
    min_y_point = (x_values.iloc[np.argmin(y_values)], np.min(y_values))

    m_from_x = (max_x_point[1] - min_x_point[1]) / (max_x_point[0] - min_x_point[0])
    m_from_y = (max_y_point[1] - min_y_point[1]) / (max_y_point[0] - min_y_point[0])
    m_guess = (m_from_x + m_from_y) / 2

    c_guess = y_values.iloc[np.argmin(np.abs(x_values))]

    return [m_guess, c_guess]


def calc_rectangular_hyperbola_bounds(x_values, y_values):
    """
    Reasonable bounds for rectangular hyperbola fitting.

    Parameters
    ----------
    Args:
        x_values (array-like): Independent variable values.
        y_values (array-like): Dependent variable values.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) for [max, k].
    """
    y_max = float(np.max(y_values))
    y_min = float(np.min(y_values))

    # Bounds for max parameter
    # Allow for noise by letting lower be ~10% of max observed,
    # but if everything is tiny, allow negatives to avoid solver stuckness.
    max_lower = y_min - 10 * (y_max - y_min)
    max_upper = y_max + 10 * (y_max - y_min)

    # Bounds for k parameter
    k_lower = float(np.min(x_values))
    k_upper = float(np.max(x_values))

    lower_bounds = [max_lower, k_lower]
    upper_bounds = [max_upper, k_upper]

    return (lower_bounds, upper_bounds)


def get_model_info(model_str, x_values=None, y_values=None):
    model_info = {}

    if model_str == "3 parameter logistic":
        model_info["parameter_names"] = ["Emax", "Emin", "EC50"]
        model_info["model_function"] = logistic_3_parameter_not_logged
        model_info["logged_x"] = False

        if (x_values is None) or (y_values is None):
            model_info["initial_guess_values"] = None
            model_info["bounds"] = None
        else:
            model_info["initial_guess_values"] = calc_initial_logistic_guess(
                x_values, y_values
            )
            model_info["bounds"] = calc_logistic_bounds(x_values, y_values)

    elif model_str == "4 parameter logistic":
        model_info["parameter_names"] = ["Emax", "Emin", "EC50", "Hillslope"]
        model_info["model_function"] = logistic_4_parameter_not_logged
        model_info["logged_x"] = False

        if (x_values is None) or (y_values is None):
            model_info["initial_guess_values"] = None
        else:
            initial_guess_values = calc_initial_logistic_guess(x_values, y_values)
            initial_guess_values.append(1)
            bounds_lower, bounds_upper = calc_logistic_bounds(x_values, y_values)
            bounds_lower.append(-np.inf)
            bounds_upper.append(np.inf)
            model_info["initial_guess_values"] = initial_guess_values
            model_info["bounds"] = (bounds_lower, bounds_upper)

    elif model_str == "rectangular_hyperbola":
        model_info["parameter_names"] = ["max", "k"]
        model_info["model_function"] = rectangular_hyperbola

        if (x_values is None) or (y_values is None):
            model_info["initial_guess_values"] = None
            model_info["bounds"] = None
        else:
            model_info["initial_guess_values"] = (
                calc_initial_rectangular_hyperbola_guess(x_values, y_values)
            )
            model_info["bounds"] = calc_rectangular_hyperbola_bounds(x_values, y_values)

    elif model_str == "straight_line":
        model_info["parameter_names"] = ["gradient", "y-intercept"]
        model_info["model_function"] = straight_line

        if (x_values is None) or (y_values is None):
            model_info["initial_guess_values"] = None
            model_info["bounds"] = None
        else:
            model_info["initial_guess_values"] = calc_initial_straight_line_guess(
                x_values, y_values
            )
            model_info["bounds"] = (-np.inf, np.inf)

    else:
        raise ValueError("Model information not available")

    return model_info


def curve_fit_with_error_handling(
    model_function, x_data, y_data, p0=None, maxfev=None, bounds=(-np.inf, np.inf)
):
    try:
        # Try fitting the curve
        params, params_covariance = curve_fit(
            model_function, x_data, y_data, p0=p0, maxfev=maxfev, bounds=bounds
        )
        return params, params_covariance
    except Exception as e:  # catch RuntimeError, ValueError, etc.
        logger.warning(f"[fit error] {e}")
        return None, None


def fit_curve_to_data(
    x_values, y_values, model=None, report_fit_summary=None, use_bounds=None
):
    """
    Fits a curve to the given x and y data, based on the model given.
    The default model is a 3-parameter logistic equation, without logged
    concentrations.

    The function applies nonlinear curve fitting to estimate parameters for a
    model. It currently supports both 3- and 4-parameter logistic models and
    includes error handling for failed curve fits.

    Args:
        x_values (array-like): Independent variable values.
        y_values (array-like): Dependent variable values.
        model (str, optional): The model to use. Options:
            - '3 parameter logistic' (default)
            - '4 parameter logistic'

    Returns:
        dict: Estimated parameters, their standard deviations, and covariances between parameters.
    """

    # Model error handling
    if model is None:
        model = "3 parameter logistic"

    available_models = [
        "3 parameter logistic",
        "4 parameter logistic",
        "rectangular_hyperbola",
        "straight_line",
    ]

    if model not in available_models:
        raise ValueError(f"Please supply model from {available_models}")

    # Extracting model info
    model_info = get_model_info(model, x_values, y_values)
    model_function = model_info["model_function"]
    initial_guess_values = model_info["initial_guess_values"]
    parameter_names = model_info["parameter_names"]
    if use_bounds:
        bounds = model_info["bounds"]
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]
    else:
        bounds = (-np.inf, np.inf)
        lower_bounds = []
        upper_bounds = []

    params, params_covariance = curve_fit_with_error_handling(
        model_function,
        x_values,
        y_values,
        p0=initial_guess_values,
        bounds=bounds,
        maxfev=10000,
    )
    if report_fit_summary:
        # Rows = parameters, columns = each stage of the fit. Bounds columns are only
        # included when bounds were actually used (otherwise the lists are empty).
        summary = {
            "Initial Guess": initial_guess_values,
            "Fit Result": (
                params
                if params is not None
                else ["curve fitting failed"] * len(parameter_names)
            ),
        }
        if use_bounds:
            summary["Lower Bound"] = lower_bounds
            summary["Upper Bound"] = upper_bounds
        summary_df = pd.DataFrame(summary, index=parameter_names)
        logger.info("Fit summary:\n%s", summary_df)

    # Handling error of unsuccessful curve fit
    if params is None or params_covariance is None:
        # Fill with NaN for clarity and filterability
        result = {}
        result.update({name: float("nan") for name in parameter_names})
        result.update({f"{name}_stdev": float("nan") for name in parameter_names})
        result.update(
            {
                f"Cov({name1}, {name2})": float("nan")
                for i, name1 in enumerate(parameter_names)
                for j, name2 in enumerate(parameter_names)
                if i < j
            }
        )
        result["fit_status"] = "failed"
        return result

    # Constructing function output of successful curve fit
    # Fitted Parameters
    params_dict = dict(zip(parameter_names, params))

    # The diagonals of the covariance matrix are the variances, so sqrt is the std_dev
    parameter_names_stdev = [f"{element}_stdev" for element in parameter_names]
    params_std_dev_dict = dict(
        zip(parameter_names_stdev, np.sqrt(np.diag(params_covariance)))
    )

    # We can unpack the parameters and their errors together:
    full_params_dict = {**params_dict, **params_std_dev_dict}

    # We can now populate the dictionary with covariances.
    # The covariance matrix is symmetric (covariance is commutative), so we only take
    # the upper triangle (i < j); the diagonal is the variance, already stored as stdev.
    for i, name1 in enumerate(parameter_names):
        for j, name2 in enumerate(parameter_names):
            if i < j:
                full_params_dict[f"Cov({name1}, {name2})"] = params_covariance[i, j]

    return full_params_dict


def auto_auc(
    df,
    time_column=TIME_MINUTES_COLUMN,
    time_column2=TIME_SECONDS_COLUMN,
    value_column=OUTPUT_VALUE_NAME,
    output_type=OUTPUT_TYPE_NAME,
):
    """
    Calculate the area under the curve (AUC) for each combination of simulation and output type.

    Parameters:
    - df (pd.DataFrame): The simulation dataframe in long format.
    - time_column (str): Name of the column representing time (either in seconds or minutes). Default is TIME_SECONDS_COLUMN.
    - value_column (str): Name of the column representing the simulation output. Default is OUTPUT_VALUE_NAME.
    - output_type (str): Name of the column representing the type of output (e.g., which species or variable). Default is OUTPUT_TYPE_NAME.

    Returns:
    - pd.DataFrame: A dataframe with each row representing a unique combination of simulation and output type
                    and an associated AUC value.
    """

    if df.empty:
        raise ValueError("Input DataFrame to auto_auc is empty.")

    # List of columns to group by (all columns except time and value columns)
    grouping_columns = list(df.columns)
    for unwanted_column in [time_column, time_column2, value_column]:
        grouping_columns.remove(unwanted_column)

    # Storing results for each combination of simulation and output type
    results = []

    # Iterate over each unique combination of simulation conditions and output type
    for simulation_conditions, specific_simulation in df.groupby(grouping_columns):
        # Extracting time and value data for AUC calculation
        x = specific_simulation[time_column].to_numpy()
        y = specific_simulation[value_column].to_numpy()
        auc_value = scipy.integrate.trapezoid(y, x)

        # Constructing a dictionary to hold simulation conditions, output type, and AUC
        simulation_info = dict(zip(grouping_columns, simulation_conditions))
        simulation_info["AUC"] = auc_value
        simulation_info["n_timepoints"] = len(x)

        # Appending the result to our results list
        results.append(simulation_info)

    # Converting our results list into a dataframe
    result_df = pd.DataFrame(results)
    logger.debug(f"the result_df after auc calculation is: \n{result_df}")

    # Ordering the columns to match the input dataframe
    ordered_columns = grouping_columns + ["AUC"] + ["n_timepoints"]
    result_df = result_df[ordered_columns]

    return result_df
