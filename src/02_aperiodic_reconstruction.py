"""
Aperiodic signal reconstruction using FOOOF.
"""

import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from fooof import FOOOF

from scipy.fftpack import fft, ifft

from pandarallel import pandarallel


pandarallel.initialize(progress_bar=True, use_memory_fs=False)


def ap_reconstructed_signal(data, overlap, duration):
    """
    Reconstruct EEG signals using the aperiodic component estimated by FOOOF.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing MNE Raw objects and metadata.
    overlap : float
        Overlap between epochs in seconds.
    duration : float
        Epoch length in seconds.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with reconstructed signals.
    """

    f_min = 0.7
    f_max = 58

    # Epoching and FFT computation
    data["epochs"] = data["mne"].parallel_apply(
        lambda raw: mne.make_fixed_length_epochs(
            raw,
            duration=duration,
            overlap=overlap,
            preload=False
        )
    )

    data["psds"] = data["epochs"].parallel_apply(
        lambda epochs: fft(
            epochs.get_data(picks="eeg"),
            axis=2
        )
    )

    del data["mne"]
    del data["epochs"]
    gc.collect()

    def aperiodic_signal(psds, sfreq):
        """
        Replace oscillatory spectrum with aperiodic spectrum and reconstruct signal.
        """

        n_samples = psds.shape[0]

        freqs = np.fft.fftfreq(
            n_samples,
            1 / sfreq
        )[: (n_samples // 2) + 1]

        # Keep only positive frequencies (required by FOOOF)
        psds = psds[: (n_samples // 2) + 1]

        phases = np.angle(psds)
        power = np.abs(psds) ** 2

        fm = FOOOF(
            aperiodic_mode="fixed",
            max_n_peaks=7,
            min_peak_height=0.3,
            peak_width_limits=[3, 40]
        )

        fm.fit(
            freqs,
            power,
            freq_range=[f_min, f_max]
        )

        aperiodic_spectrum = fm._spectrum_peak_rm

        # Replace power spectrum in fitted range
        freq_indices = np.where(
            (freqs >= f_min) & (freqs <= f_max)
        )[0]

        power[freq_indices] = 10 ** aperiodic_spectrum

        spectrum_reconstructed = (
            np.sqrt(power) * np.exp(1j * phases)
        )

        # Hermitian symmetry for inverse FFT
        freq_neg = np.conj(
            spectrum_reconstructed[1:-1][::-1]
        )

        spectrum = np.concatenate(
            [spectrum_reconstructed, freq_neg]
        )

        ifft_data = ifft(spectrum)

        return ifft_data.real

    def parallel_aperiodic(row):
        """
        Apply aperiodic reconstruction to all channels and epochs.
        """

        return np.apply_along_axis(
            aperiodic_signal,
            2,
            row.psds,
            row.sfreq
        )

    # Aperiodic reconstruction in frequency domain
    data["reconstructed_epochs"] = data.apply(
        parallel_aperiodic,
        axis=1
    )

    del data["psds"]
    gc.collect()

    def reconstruct_signal(row):
        """
        Reconstruct continuous signal from overlapping epochs.
        """

        aperiodic_signal_epochs = row.reconstructed_epochs

        artifact_points = 5
        overlap_samples = overlap * round(row.sfreq)

        split_epochs = np.split(
            aperiodic_signal_epochs,
            [artifact_points, overlap_samples, -artifact_points],
            axis=-1
        )

        # Zero-padding artifact regions
        split_epochs[1] = np.pad(
            split_epochs[1],
            ((0, 0), (0, 0), (artifact_points, 0)),
            mode="constant"
        )

        split_epochs[2] = np.pad(
            split_epochs[2],
            ((0, 0), (0, 0), (0, artifact_points)),
            mode="constant"
        )

        shape_original = list(split_epochs[1].shape)
        shape_original[0] = shape_original[0] * 2

        # Interleave overlapping segments
        epochs_ordered = (
            np.stack((split_epochs[1], split_epochs[2]))
            .swapaxes(0, 1)
            .reshape(shape_original)
        )

        # Average overlapping regions
        overlap_mean = np.stack(
            np.split(
                epochs_ordered[1:-1],
                (shape_original[0] - 2) // 2
            )
        ).sum(axis=1)

        overlap_mean[:, :, artifact_points:-artifact_points] /= 2

        reconstructed_signal = np.hstack((
            epochs_ordered[0],
            overlap_mean.reshape(len(row.e_label), -1),
            epochs_ordered[-1]
        ))

        return reconstructed_signal

    # Time-domain reconstruction
    data["reconstructed_signal"] = data.parallel_apply(
        reconstruct_signal,
        axis=1
    )

    del data["reconstructed_epochs"]
    gc.collect()

    data.to_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/reconstructed_signals.pkl"
    )

    return data


if __name__ == "__main__":

    data = pd.read_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/data.pkl"
    )

    data.drop(columns=["epochs", "psds"], inplace=True)

    gc.collect()

    processed_data = ap_reconstructed_signal(
        data,
        overlap=1,
        duration=2
    )
