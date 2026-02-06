"""
EEG preprocessing and segmentation pipeline for TUSZ-BIDS,
designed for large-scale parallel processing of subjects and sessions.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne import Annotations
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

from scipy.fft import fft
import dask.dataframe as dd


warnings.filterwarnings("ignore")


def event_annotations(annotations, time):
    """
    Generate event labels for time windows based on seizure annotations.

    Parameters
    ----------
    annotations : mne.Annotations
        Annotations containing seizure onset and duration.
    time : np.ndarray
        Array of time points corresponding to window starts.

    Returns
    -------
    np.ndarray
        Array of integer labels indicating event type for each time window.
    """

    sz_begin = []
    sz_final = []

    for annot in annotations:
        sz_begin.append(annot["onset"])
        sz_final.append(annot["onset"] + annot["duration"])

    # Collect time points that fall inside seizure intervals
    interictal_times = np.concatenate([
        time[(time > start) & (time < end)]
        for start, end in zip(sz_begin, sz_final)
    ])

    # Initial labeling based on first and last seizure
    event_labels = np.where(
        np.logical_not(time < sz_begin[0]),
        time,
        2
    )

    event_labels = np.where(
        np.logical_not(event_labels > sz_final[-1]),
        event_labels,
        3
    )

    interictal_times = np.array(interictal_times)

    # Mark seizure-related windows
    event_labels = np.where(
        np.logical_not(np.isin(event_labels, interictal_times)),
        event_labels,
        0
    )

    # Final binarization
    event_labels = np.where(
        np.isin(event_labels, [0, 2, 3]),
        event_labels,
        1
    )

    return event_labels


def process_run(subject, session, run, root, duration, overlap):
    """
    Process a single run from one subject and session.

    This function:
    - Loads BIDS data
    - Renames channels
    - Loads event annotations
    - Filters signals
    - Creates fixed-length epochs
    - Generates event labels

    Parameters
    ----------
    subject : str
        Subject identifier.
    session : str or int
        Session identifier.
    run : int
        Run number.
    root : str
        Root directory of BIDS dataset.
    duration : float
        Epoch duration in seconds.
    overlap : float
        Overlap between epochs in seconds.

    Returns
    -------
    dict or None
        Dictionary with processed data, or None if loading fails.
    """

    try:
        bids_path = BIDSPath(
            root=root,
            subject=subject,
            session=str(session),
            task="szMonitoring",
            run="0" + str(run)
        )

        raw = read_raw_bids(bids_path=bids_path, verbose=False)

        # Rename channels for consistency
        old_names = raw.ch_names
        new_names = {
            name: name.replace("-Avg", "").title()
            for name in old_names
        }
        raw.rename_channels(new_names)

        # Load event information
        events = pd.read_csv(
            str(bids_path)[:-7] + "events.tsv",
            sep="\t"
        )

        annotations = Annotations(
            onset=events["onset"].values,
            duration=events["duration"].values,
            description=events["eventType"].values
        )

        raw.set_annotations(annotations)
        annotations = raw.annotations

        # Filtering
        raw.load_data()
        raw.filter(l_freq=0.5, h_freq=100, verbose=False)
        raw.notch_filter(freqs=60, notch_widths=2, picks="data")

        # EEG channel selection and montage
        channels = raw.copy().pick_types(eeg=True).ch_names

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)

        # Crop background recordings
        if np.isin("bckg", annotations.description):
            raw.crop(tmin=20)

        # Fixed-length segmentation
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=duration,
            overlap=overlap,
            preload=False
        )

        duration_segment = int(raw.n_times / raw.info["sfreq"])

        time = np.arange(
            0,
            duration_segment - duration + 0.1,
            overlap
        )

        # Event labeling
        if np.isin("sz", annotations.description):
            events_array = event_annotations(
                annotations=annotations,
                time=time
            )

        elif np.isin("bckg", annotations.description):
            events_array = np.tile(4, time.shape)

        return {
            "name": subject,
            "mne": raw.copy(),
            "epochs": epochs,
            "events_array": events_array,
            "e_label": channels,
            "session_run": str(session) + "-" + str(run),
            "sfreq": raw.info["sfreq"]
        }

    except (FileNotFoundError, ValueError):
        return None


def process_subject_session(subject, session, root, duration, overlap):
    """
    Process all runs for one subject and one session.

    Parameters
    ----------
    subject : str
        Subject ID.
    session : str
        Session ID.
    root : str
        Dataset root path.
    duration : float
        Epoch duration.
    overlap : float
        Epoch overlap.

    Returns
    -------
    list
        List of processed runs.
    """

    run_data = []
    run = 1

    while True:
        result = process_run(
            subject,
            session,
            run,
            root,
            duration,
            overlap
        )

        if result is None:
            break

        run_data.append(result)
        run += 1

    return run_data


def process_subject(subject, sessions, root, duration, overlap):
    """
    Process all sessions for one subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    sessions : list
        List of session identifiers.
    root : str
        Dataset root.
    duration : float
        Epoch duration.
    overlap : float
        Epoch overlap.

    Returns
    -------
    list
        List of processed runs.
    """

    all_runs = []

    for session in sessions:
        runs = process_subject_session(
            subject,
            session,
            root,
            duration,
            overlap
        )

        all_runs.extend(runs)

    return all_runs


def main():
    """
    Main execution function.

    Loads subject list, parallelizes processing,
    computes PSDs, and saves output files.
    """

    root = "/storage5/ame.lia/TUSZ-BIDS1"

    duration = 4
    overlap = 2

    f_min = 0.7
    f_max = 60

    # Detect available subjects
    subjects = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    data = pd.DataFrame(
        columns=[
            "name",
            "mne",
            "events_array",
            "e_label",
            "session_run",
            "sfreq",
            "epochs"
        ]
    )

    # Map subjects to sessions
    subject_sessions = {
        subject[4:]: get_entity_vals(
            os.path.join(root, subject),
            "session"
        )
        for subject in subjects
    }

    # Parallel processing by subject
    with ProcessPoolExecutor() as executor:

        futures = [
            executor.submit(
                process_subject,
                subj,
                sess_list,
                root,
                duration,
                overlap
            )
            for subj, sess_list in subject_sessions.items()
        ]

        for future in as_completed(futures):

            results = future.result()

            for result in results:
                data.loc[len(data)] = result

    # Power spectral density computation
    data["psds"] = data["epochs"].apply(
        lambda epochs: epochs.compute_psd(
            method="multitaper",
            fmin=f_min,
            fmax=f_max
        ).get_data(return_freqs=False)
    )

    # Save outputs
    data.to_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/data.pkl"
    )

    data["events_array"].to_pickle(
        "/storage5/ame.lia/TUSZ-BIDS1/events_array.pkl"
    )


if __name__ == "__main__":
    main()
