"""
Parallel computation of time-series nonlinear measures
"""

import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd

from zlib import compress
from tqdm import tqdm

import EntropyHub as EH
import nolds

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import skew, kurtosis


# Output dimensionality for each feature
OUTPUT_SHAPES = {
    "dfa": 1,
    "lz_complexity": 1,
    "permen": 5,
    "aproxen": 4,
    "sampeen": 4
}


def correlation_func(timeseries, tau=60):
    """
    Estimate time delay using mutual information decay.
    """

    for lag in range(1, tau + 1):

        timeseries_x = timeseries[:-lag]
        timeseries_y = timeseries[lag:]

        r = mutual_info_regression(
            timeseries_x.reshape(-1, 1),
            timeseries_y
        )

        if r < 1 / np.e:
            break

    return int(lag)


def lz_complexity(timeseries):
    """
    Compute Lempel-Ziv complexity of a binarized time series.
    """

    mean = np.mean(timeseries)

    coarse_grained = "".join(
        "1" if x > mean else "0"
        for x in timeseries
    )

    unique_subsecuences = compress(
        bytes(coarse_grained, "utf-8")
    )

    complexity = (
        len(unique_subsecuences)
        * np.log2(len(coarse_grained))
        / len(coarse_grained)
    )

    return complexity


def dfa(timeseries, plot=False, all_data=False):
    """
    Compute detrended fluctuation analysis exponent.
    """

    return nolds.dfa(
        timeseries,
        overlap=True,
        order=1
    )


def aproxen(timeseries, m=5, num_std=0.2):
    """
    Compute approximate entropy.
    """

    tau = correlation_func(timeseries)

    r = num_std * np.std(timeseries)

    aprox, _ = EH.ApEn(
        timeseries,
        m,
        tau,
        r
    )

    return aprox[-4:]


def sampeen(timeseries, m=5, num_std=0.2):
    """
    Compute sample entropy.
    """

    tau = correlation_func(timeseries)

    r = num_std * np.std(timeseries)

    sampen, _, _ = EH.SampEn(
        timeseries,
        m=m,
        tau=tau,
        r=r
    )

    return sampen[-4:]


def permen(timeseries, m=6, num_std=0.2):
    """
    Compute permutation entropy.
    """

    tau = correlation_func(timeseries)

    permen, Pnorm, cPE = EH.PermEn(
        timeseries,
        m=m,
        tau=tau
    )

    return permen[-5:]



def compute_techniques(args):
    """
    Apply one complexity function to one time series.
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
    Compute nonlinear measures for all epochs and channels.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing original and reconstructed ap epochs.
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

                    ts = epochs_arr[epoch_i, channel_j, :]

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
        "/storage5/ame.lia/TUSZ-BIDS1/complexity.pkl"
    )


if __name__ == "__main__":

    functions = [
        dfa,
        lz_complexity,
        permen,
        aproxen,
        sampeen
    ]

    all_signals = pd.read_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/all_signals.pkl"
    )

    complexity(all_signals, functions)
