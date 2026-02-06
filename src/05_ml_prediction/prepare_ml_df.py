"""
Cleans feature tables, assigns anatomical labels,
balances background segments, and aggregates features by electrode.
"""

import gc
import re
import random

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype


random.seed(42)


def subsample_bckgd_trim_extremes_per_run(
    df,
    target_bg_per_run,
    id_col="name",
    run_col="session_run",
    time_col="time",
    cond_col="events_array",
    bg_label="bckgd",
    keep_non_bg=True,
    trim_mode="alternate",
    shuffle=True,
    seed=42,
    reset_index=True,
    verbose=True,
):
    """
    Limit the number of background windows per run by trimming
    from the temporal extremes.
    """

    if target_bg_per_run <= 0:
        raise ValueError("target_bg_per_run must be > 0")

    needed = [id_col, run_col, time_col, cond_col]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if verbose:
        print("Before subsampling:")
        print(df[cond_col].value_counts())

    bg = df[df[cond_col] == bg_label].copy()
    non_bg = df[df[cond_col] != bg_label].copy()

    keep_idx = []

    bg = bg.sort_values([id_col, run_col, time_col])

    for (subj, run), g in bg.groupby([id_col, run_col], sort=False):

        n = len(g)

        if n <= target_bg_per_run:
            keep_idx.append(g.index.to_numpy())
            continue

        remove = n - target_bg_per_run

        if trim_mode == "alternate":

            idx = g.index.to_numpy()
            left = 0
            right = n - 1

            drop_mask = np.zeros(n, dtype=bool)

            for k in range(remove):

                if k % 2 == 0:
                    drop_mask[left] = True
                    left += 1
                else:
                    drop_mask[right] = True
                    right -= 1

            kept = idx[~drop_mask]
            keep_idx.append(kept)

        elif trim_mode == "both_sides_then_tail":

            cut_left = remove // 2
            cut_right = remove - cut_left

            kept = g.iloc[
                cut_left : n - cut_right
            ].index.to_numpy()

            keep_idx.append(kept)

        else:
            raise ValueError("Invalid trim_mode")

    bg_down = (
        bg.loc[np.concatenate(keep_idx)]
        if keep_idx
        else bg.iloc[0:0]
    )

    out = (
        pd.concat([bg_down, non_bg], axis=0)
        if keep_non_bg
        else bg_down
    )

    if shuffle:
        out = out.sample(frac=1, random_state=seed)

    if reset_index:
        out = out.reset_index(drop=True)

    if verbose:
        print("\nAfter subsampling:")
        print(out[cond_col].value_counts())

    return out


def limit_bckgd_runs_per_subject(
    df,
    max_bg_runs_per_subject,
    id_col="name",
    run_col="session_run",
    cond_col="events_array",
    bg_label="bckgd",
    strategy="random",
    preictal_label="preictal",
    seed=42,
    verbose=True,
):
    """
    Limit the number of runs contributing background data per subject.
    """

    if max_bg_runs_per_subject <= 0:
        raise ValueError("max_bg_runs_per_subject must be > 0")

    df_out = df.copy()

    bg = df_out[df_out[cond_col] == bg_label].copy()

    if bg.empty:
        if verbose:
            print("No background data found.")
        return df_out

    rng = np.random.default_rng(seed)

    if strategy == "most_preictal":

        pre_counts = (
            df_out[df_out[cond_col] == preictal_label]
            .groupby([id_col, run_col])
            .size()
        )

    else:
        pre_counts = None

    drop_idx = []

    for subj, bg_sub in bg.groupby(id_col, sort=False):

        runs = bg_sub[run_col].unique()

        if len(runs) <= max_bg_runs_per_subject:
            continue

        if strategy in ("largest", "smallest", "random"):

            bg_sizes = bg_sub.groupby(run_col).size()

            if strategy == "largest":

                keep_runs = (
                    bg_sizes
                    .sort_values(ascending=False)
                    .head(max_bg_runs_per_subject)
                    .index
                    .tolist()
                )

            elif strategy == "smallest":

                keep_runs = (
                    bg_sizes
                    .sort_values()
                    .head(max_bg_runs_per_subject)
                    .index
                    .tolist()
                )

            else:

                keep_runs = rng.choice(
                    runs,
                    size=max_bg_runs_per_subject,
                    replace=False
                ).tolist()

        elif strategy == "most_preictal":

            bg_sizes = bg_sub.groupby(run_col).size()
            scores = []

            for r in runs:

                p = int(pre_counts.get((subj, r), 0))
                b = int(bg_sizes.get(r, 0))

                scores.append((r, p, b))

            scores.sort(
                key=lambda t: (t[1], t[2]),
                reverse=True
            )

            keep_runs = [
                t[0] for t in scores[:max_bg_runs_per_subject]
            ]

        else:
            raise ValueError("Invalid strategy")

        to_drop = bg_sub[
            ~bg_sub[run_col].isin(keep_runs)
        ].index

        drop_idx.append(to_drop.to_numpy())

    if drop_idx:

        drop_idx = np.concatenate(drop_idx)

        df_out = df_out.drop(index=drop_idx)

    if verbose:

        print(
            f"Background runs limited: max={max_bg_runs_per_subject}, "
            f"strategy={strategy}"
        )

    return df_out


# Load feature table
complexity = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/all_proxies.pkl"
)

complexity = complexity.reset_index(drop=True)

complexity.drop(columns=["sfreq"], inplace=True)


# Map event codes to labels
period = {
    0: "sz",
    1: "interictal",
    2: "preictal",
    3: "postictal",
    4: "bckgd",
}

complexity["events_array"] = complexity["events_array"].map(period)

complexity.loc[
    (complexity.time > -5.0)
    & (complexity.events_array == "preictal"),
    "events_array"
] = "close_preictal"

complexity = complexity[
    ~complexity["events_array"].isin(
        ["interictal", "postictal", "close_preictal"]
    )
]


# Channel to lobe mapping
_ELECTRODE_LOBE = {

    "Fp1": "Frontal",
    "Fp2": "Frontal",
    "Fz": "Frontal",
    "F3": "Frontal",
    "F4": "Frontal",
    "F7": "Frontal",
    "F8": "Frontal",

    "T3": "Temporal",
    "T4": "Temporal",
    "T5": "Temporal",
    "T6": "Temporal",
    "T7": "Temporal",
    "T8": "Temporal",

    "Cz": "Central",
    "C3": "Central",
    "C4": "Central",

    "Pz": "Parietal",
    "P3": "Parietal",
    "P4": "Parietal",

    "O1": "Occipital",
    "O2": "Occipital",
}


def _hemi(ch):

    if ch.lower().endswith("z"):
        return "Midline"

    m = re.search(r"(\d+)$", ch)

    if not m:
        return "Unknown"

    return "Left" if int(m.group(1)) % 2 == 1 else "Right"


def add_topography(df):

    df = df.copy()

    df["lobe"] = (
        df["e_label"]
        .map(_ELECTRODE_LOBE)
        .fillna("Unknown")
    )

    df["hemi"] = df["e_label"].astype(str).map(_hemi)

    return df


complexity = add_topography(complexity)


# Handle invalid values
complexity.replace([np.inf, -np.inf], np.nan, inplace=True)

for col in ["og_sampeen_5", "og_sampeen_4", "ap_sampeen_5"]:

    complexity[col].fillna(
        complexity[col].median(),
        inplace=True
    )


# Aggregate by electrode
info_cols = ["events_array", "name", "session_run", "time"]

all_features = list(
    set(complexity.columns) - set(info_cols)
)

pivot_idx = ["name", "session_run", "events_array", "time"]

MEASURE_COLS = [
    c for c in all_features
    if is_numeric_dtype(complexity[c])
]


wide_electrode = complexity.pivot_table(
    index=pivot_idx,
    columns="e_label",
    values=MEASURE_COLS,
    aggfunc="mean"
)

wide_electrode.columns = [
    f"{feat}_{e}_{func}"
    for (func, feat, e)
    in wide_electrode.columns.to_list()
]

wide_electrode.reset_index(inplace=True)


# Balance background data
wide_electrode = limit_bckgd_runs_per_subject(
    wide_electrode,
    max_bg_runs_per_subject=6,
    strategy="largest",
    seed=42,
    verbose=True
)

wide_electrode = subsample_bckgd_trim_extremes_per_run(
    wide_electrode,
    target_bg_per_run=250,
    trim_mode="alternate",
    seed=42,
    verbose=True
)


# Save final table
wide_electrode.to_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/wide_electrode_250_6.pkl"
)

gc.collect()
