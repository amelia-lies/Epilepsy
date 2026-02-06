"""
Parallel computation of basic statistical moments
"""

import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis


# Output dimensionality for each feature
OUTPUT_SHAPES = {
    "mean_value": 1,
    "variance": 1,
    "skewness": 1,
    "kurtosis": 1
}


def skewness(ts):
    """
    Compute unbiased skewness, ignoring NaNs.
    """

    val = skew(
        ts,
        bias=False,
        nan_policy="omit"
    )

    return float(
        0.0 if not np.isfinite(val) else val
    )


def kurtosis(ts):
    """
    Compute Fisher kurtosis excess, ignoring NaNs.
    """

    val = kurtosis(
        ts,
        fisher=True,
        bias=False,
        nan_policy="omit"
    )

    if not np.isfinite(val):
        return float(-3.0)

    return float(val)


def mean_value(timeseries):
    """
    Compute mean ignoring NaNs.
    """

    return float(
        np.nanmean(timeseries)
    )


def variance(timeseries):
    """
    Compute variance ignoring NaNs.
    """

    return float(
        np.nanvar(timeseries, ddof=0)
    )


def compute_techniques(args):
    """
    Apply one statistical function to one time series.
    """

    ts_idx, func_name, ts = args

    func = globals()[func_name]

    try:
        result = func(ts)
        return (ts_idx, func_name, result, None)

    except Exception as e:
        return (ts_idx, func_name, None, str(e))


def complexity(data, functions):
    """
    Compute statistical moments for all epochs and channels.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing original and reconstructed epochs.
    functions : list
        List of feature functions.
    """

    func_names = [
        func.__name__
        for func in functions
    ]

    tasks = []

    # Build processing tasks
    for idx, row in data.iterrows():

        for epoch_type in ["og_epochs", "ap_epochs"]:

            epochs_arr = row[epoch_type]

            for epoch_i in range(epochs_arr.shape[0]):
                for channel_j in range(epochs_arr.shape[1]):

                    ts = epochs_arr[
                        epoch_i,
                        channel_j,
                        :
                    ]

                    ts_idx = (
                        idx,
                        epoch_type,
                        epoch_i,
                        channel_j
                    )

                    for name in func_names:
                        tasks.append(
                            (ts_idx, name, ts)
                        )

    results = {}

    # Parallel processing
    with mp.Pool(
        processes=mp.cpu_count()
    ) as pool:

        for ts_idx, func_name, result_val, error in pool.imap(
            compute_techniques,
            tasks
        ):

            if error:
                print(
                    f"Error in {func_name} for {ts_idx}: {error}",
                    file=sys.stderr
                )
                continue

            if ts_idx not in results:
                results[ts_idx] = {}

            results[ts_idx][func_name] = result_val

    # Initialize output columns
    for func_name in func_names:

        data[f"og_{func_name}"] = [None] * len(data)
        data[f"ap_{func_name}"] = [None] * len(data)

    # Fill DataFrame with results
    for ts_idx, func_results in results.items():

        idx, epoch_type, epoch_i, channel_j = ts_idx

        prefix = (
            "og" if epoch_type == "og_epochs"
            else "ap"
        )

        for func_name, val in func_results.items():

            col_name = f"{prefix}_{func_name}"

            cell = data.at[idx, col_name]

            if cell is None:

                n_epochs, n_channels = data.at[
                    idx,
                    epoch_type
                ].shape[:2]

                out_shape = OUTPUT_SHAPES[func_name]

                cell = np.empty(
                    (n_epochs, n_channels, out_shape),
                    dtype=np.float64
                )

                data.at[idx, col_name] = cell

            data.at[idx, col_name][
                epoch_i,
                channel_j
            ] = val

    data.drop(
        columns=["og_epochs", "ap_epochs"],
        inplace=True
    )

    data.to_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/momentos.pkl"
    )


if __name__ == "__main__":

    functions = [
        mean_value,
        variance,
        kurtosis,
        skewness
    ]

    all_signals = pd.read_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/all_signals.pkl"
    )
    
    complexity(all_signals, functions)
