"""
Generation of comparable epoch representations for original and
aperiodic-reconstructed EEG signals.
"""

import gc

import numpy as np
import pandas as pd

import mne
from pandarallel import pandarallel


pandarallel.initialize(progress_bar=True)


# Load reconstructed signals and original dataset
reconstruct_signal = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/reconstructed_signals.pkl"
)

data = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/data.pkl"
)

# Merge original MNE objects and epochs
reconstruct_signal = pd.concat(
    [
        reconstruct_signal,
        data.loc[:, ["mne", "epochs"]]
    ],
    axis=1
)


def ap_mne(row, window_size, overlap):
    """
    Convert reconstructed continuous signal into MNE epochs.
    """

    info = row.mne.copy().pick_types(eeg=True).info

    raw = mne.io.RawArray(
        row.reconstructed_signal,
        info
    )

    # Free memory from large objects
    reconstruct_signal.loc[
        reconstruct_signal.index == row.name,
        "reconstructed_signal"
    ] = np.nan

    reconstruct_signal.loc[
        reconstruct_signal.index == row.name,
        "mne"
    ] = np.nan

    gc.collect()

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=window_size,
        overlap=overlap
    )

    return epochs.get_data(picks="eeg")


# Epoch reconstructed signals
reconstruct_signal["ap_epochs"] = reconstruct_signal.parallel_apply(
    ap_mne,
    args=(4, 2),
    axis=1
)

del reconstruct_signal["reconstructed_signal"]
del reconstruct_signal["mne"]
gc.collect()


# Extract original epochs
reconstruct_signal["og_epochs"] = reconstruct_signal["epochs"].parallel_apply(
    lambda epochs: epochs.get_data(picks="eeg")
)

del reconstruct_signal["epochs"]
gc.collect()


# Report memory usage
memory_mb = reconstruct_signal.memory_usage(deep=True).sum() / 1e6
print(memory_mb, "GB")


# Save final dataset
reconstruct_signal.to_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/all_signals.pkl"
)
