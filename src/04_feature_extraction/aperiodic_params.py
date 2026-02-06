"""
Estimates offset and exponent parameters from taper power spectra
using group-level FOOOF fitting 
"""

import numpy as np
import pandas as pd

import mne
from fooof import FOOOFGroup

from pandarallel import pandarallel


pandarallel.initialize()


# Load preprocessed dataset
data = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/data.pkl"
)


def aperiodic_params(freqs, psd, f_min, f_max):
    """
    Fit FOOOFGroup model and extract offset and exponent

    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector.
    psd : np.ndarray
        Power spectra (n_spectra x n_freqs).
    f_min : float
        Minimum frequency for fitting.
    f_max : float
        Maximum frequency for fitting.

    Returns
    -------
    np.ndarray
        Aperiodic parameters (offset, exponent).
    """

    fm = FOOOFGroup(
        aperiodic_mode="fixed",
        max_n_peaks=7,
        min_peak_height=0.3,
        peak_width_limits=[3, 40]
    )

    fm.fit(
        freqs,
        psd,
        freq_range=[f_min, f_max]
    )

    aperiodic_params = fm.get_params("aperiodic_params")

    return aperiodic_params


def aperdioc_processing(row, f_min, f_max):
    """
    Compute aperiodic parameters for all epochs and channels in one row.
    """

    psds_shape = row["psds"].shape

    # Frequency vector matching PSD length
    freqs = np.linspace(
        f_min,
        f_max,
        psds_shape[2]
    )

    params = aperiodic_params(
        freqs,
        row["psds"].reshape(-1, psds_shape[2]),
        f_min,
        f_max
    )

    params = params.reshape(
        -1,
        len(row["e_label"]),
        2
    )

    return pd.Series(
        np.split(params, 2, axis=-1)
    )


def pandas_aperiodic(data):
    """
    Parallel processing.
    """

    f_min = 0.7
    f_max = 58

    subset = data.loc[
        :,
        [
            "name",
            "events_array",
            "e_label",
            "session_run",
            "sfreq",
            "psds"
        ]
    ]

    # Parallel extraction of offset and exponent
    subset[["offset", "exponent"]] = subset.parallel_apply(
        aperdioc_processing,
        axis=1,
        args=(f_min, f_max)
    )

    subset.to_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/aperiodic.pkl"
    )


if __name__ == "__main__":

    pandas_aperiodic(data)
