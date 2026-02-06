import numpy as np
import pandas as pd
import seaborn as sns
import gc


# Load datasets
complexity = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/complexity.pkl"
)

aperiodic_power = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/aperiodic.pkl"
)

moments = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/momentos.pkl"
)


# Remove unused columns
complexity.drop(
    columns=["og_permenmod", "ap_permenmod"],
    inplace=True
)


# Merge all feature sources
complexity = pd.concat(
    [
        complexity,
        aperiodic_power.loc[:, ["offset", "exponent"]],
        moments.loc[
            :,
            [
                "og_mean_value",
                "ap_mean_value",
                "og_variance",
                "ap_variance",
                "og_skewness",
                "ap_skewness"
            ]
        ]
    ],
    axis=1
)


# Columns that will be exploded
cols_to_convert = [
    "ap_dfa",
    "og_dfa",
    "ap_lz_complexity",
    "og_lz_complexity",
    "ap_permen",
    "og_permen",
    "ap_aproxen",
    "og_aproxen",
    "ap_sampeen",
    "og_sampeen",
    "exponent",
    "offset",
    "og_mean_value",
    "ap_mean_value",
    "og_variance",
    "ap_variance",
    "og_skewness",
    "ap_skewness"
]


def time_array(events_array):
    """
    Generate time vector aligned to seizure onset.
    """

    events = events_array

    try:
        event_sz = np.argwhere(events == 0)[0]
        time = np.arange(0, len(events) * 2, 2) - event_sz * 2

    except IndexError:
        time = np.arange(0, len(events) * 2, 2)

    return time


# Compute relative time
complexity["time"] = complexity["events_array"].apply(time_array)


# Explode by epochs and time
complexity = complexity.explode(
    cols_to_convert + ["time", "events_array"]
)


# Explode by channel
complexity = complexity.explode(
    cols_to_convert + ["e_label"]
)


# Explode aperiodic parameters
complexity = complexity.explode(
    ["exponent", "offset"]
)

complexity = complexity.explode(
    [
        "exponent",
        "offset",
        "ap_dfa",
        "og_dfa",
        "ap_lz_complexity",
        "og_lz_complexity",
        "og_mean_value",
        "ap_mean_value",
        "og_variance",
        "ap_variance",
        "og_skewness",
        "ap_skewness"
    ]
)


# Mapping for multivariate entropy outputs
columns_mapping = {
    "ap_permen": [
        "ap_permen_2",
        "ap_permen_3",
        "ap_permen_4",
        "ap_permen_5",
        "ap_permen_6"
    ],
    "og_permen": [
        "og_permen_2",
        "og_permen_3",
        "og_permen_4",
        "og_permen_5",
        "og_permen_6"
    ],
    "ap_aproxen": [
        "ap_aproxen_2",
        "ap_aproxen_3",
        "ap_aproxen_4",
        "ap_aproxen_5"
    ],
    "og_aproxen": [
        "og_aproxen_2",
        "og_aproxen_3",
        "og_aproxen_4",
        "og_aproxen_5"
    ],
    "ap_sampeen": [
        "ap_sampeen_2",
        "ap_sampeen_3",
        "ap_sampeen_4",
        "ap_sampeen_5"
    ],
    "og_sampeen": [
        "og_sampeen_2",
        "og_sampeen_3",
        "og_sampeen_4",
        "og_sampeen_5"
    ]
}


# Expand entropy vectors into columns
expanded_cols = []

for column, new_columns in columns_mapping.items():

    expanded_data = np.vstack(
        complexity[column].values
    )

    df_expanded = pd.DataFrame(
        expanded_data,
        columns=new_columns,
        index=complexity.index
    )

    expanded_cols.append(df_expanded)


complexity = pd.concat(
    [complexity.drop(columns=columns_mapping.keys())]
    + expanded_cols,
    axis=1
)


# Final feature list
all_feature_cols = (
    [
        "exponent",
        "offset",
        "ap_dfa",
        "og_dfa",
        "ap_lz_complexity",
        "og_lz_complexity",
        "og_mean_value",
        "ap_mean_value",
        "og_variance",
        "ap_variance",
        "og_skewness",
        "ap_skewness"
    ]
    + list(
        np.concatenate(
            list(columns_mapping.values())
        )
    )
)


# Convert all features to numeric
complexity[all_feature_cols] = complexity[
    all_feature_cols
].apply(
    pd.to_numeric,
    errors="coerce"
)


# Save final dataset
complexity.to_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/all_proxies.pkl"
)

gc.collect()
