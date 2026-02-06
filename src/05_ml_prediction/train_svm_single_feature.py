"""
Single-feature SVM training with subject-wise normalization, session-based splitting
and group-aware cross-validation.
"""

import os
import re
import json
import hashlib
import joblib

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def zscore_by_subject_using_baseline(
    df,
    id_col="name",
    cond_col="events_array",
    baseline_label="bckgd",
    info_cols=("events_array", "name", "session_run", "time", "new_class"),
    drop_subjects_without_baseline=True,
    exclude_feature_patterns=("PCA",),
    scaler_cls=StandardScaler,
    verbose=True,
):
    """
    Z-score normalization per subject using background as baseline.
    """

    df_out = df.copy()

    feats = list(df_out.columns)

    for pat in exclude_feature_patterns or ():
        feats = [c for c in feats if pat not in c]

    feats = [c for c in feats if c not in set(info_cols)]
    feats = list(dict.fromkeys(feats))

    if not feats:
        raise ValueError("No valid feature columns found.")

    non_numeric = [
        c for c in feats
        if not pd.api.types.is_numeric_dtype(df_out[c])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric features: {non_numeric[:10]}"
        )

    dropped = []

    for subj, sub in df_out.groupby(id_col, sort=False):

        baseline = sub[sub[cond_col] == baseline_label]

        if baseline.empty:
            dropped.append(subj)
            continue

        try:
            scaler = scaler_cls()
            scaler.fit(baseline[feats])

            idx = df_out.index[df_out[id_col] == subj]

            df_out.loc[idx, feats] = scaler.transform(
                df_out.loc[idx, feats]
            )

        except Exception:
            dropped.append(subj)

    if drop_subjects_without_baseline and dropped:
        df_out = df_out[
            ~df_out[id_col].isin(dropped)
        ].copy()

    if verbose:
        print(f"Features normalized: {len(feats)}")
        print(f"Dropped subjects: {len(dropped)}")

    return df_out, dropped


# ---------------------------------------------------------------------
# Temporal volatility
# ---------------------------------------------------------------------

def add_temporal_volatility_selective(
    df,
    window=5,
    min_periods=2,
    id_cols=("id", "time"),
    include_columns=None,
    eps=1e-12,
):
    """
    Add rolling mean and coefficient of variation features.
    """

    if not include_columns:
        raise ValueError("include_columns must be non-empty.")

    df = df.sort_values(list(id_cols)).copy()

    selected = list(dict.fromkeys(include_columns))

    for c in selected:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"Invalid feature: {c}")

    vol = pd.DataFrame(index=df.index)

    for subj, g in df.groupby("id", sort=False):

        X = g[selected]

        r = X.rolling(
            window=window,
            min_periods=min_periods
        )

        std = r.std(ddof=0)
        mean = r.mean().abs()

        std.columns = [f"{c}__vol{window}_std" for c in std.columns]
        mean.columns = [f"{c}__vol{window}_mean" for c in mean.columns]

        cv = std.values / (mean.values + eps)

        cv = pd.DataFrame(
            cv,
            index=std.index,
            columns=[
                c.replace(
                    f"__vol{window}_std",
                    f"__vol{window}_cv"
                )
                for c in std.columns
            ],
        )

        vol.loc[g.index, std.columns] = std.values
        vol.loc[g.index, mean.columns] = mean.values
        vol.loc[g.index, cv.columns] = cv.values

    out = pd.concat([df, vol], axis=1)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


# ---------------------------------------------------------------------
# Session splitting
# ---------------------------------------------------------------------

def split_sessions_within_name(
    df,
    name_col="name",
    session_col="session_run",
    label_col="events_array",
    keep_labels=("bckgd", "preictal"),
    test_size=0.2,
    seed=42,
    require_both_labels=True,
    balance_globally=True,
):
    """
    Session-level split preserving subject structure.
    """

    rng = np.random.default_rng(seed)

    df2 = df[df[label_col].isin(keep_labels)].copy()

    sess = df2[
        [name_col, session_col, label_col]
    ].drop_duplicates()

    counts = pd.crosstab(
        sess[name_col],
        sess[label_col]
    )

    valid = counts.index

    if require_both_labels:
        valid = counts.index[
            (counts[keep_labels[0]] > 0)
            & (counts[keep_labels[1]] > 0)
        ]

    sess = sess[sess[name_col].isin(valid)]

    if sess.empty:
        raise ValueError("No valid sessions.")

    sessions_by = {
        n: {
            lab: sess[
                (sess[name_col] == n)
                & (sess[label_col] == lab)
            ][session_col].values
            for lab in keep_labels
        }
        for n in counts.index
    }

    target = {
        lab: int(
            np.round(
                sess[label_col].value_counts()[lab]
                * test_size
            )
        )
        for lab in keep_labels
    }

    pool = {
        n: {
            lab: sessions_by[n][lab].copy()
            for lab in keep_labels
        }
        for n in counts.index
    }

    test_sel = {lab: [] for lab in keep_labels}

    def take_one(n, lab):

        arr = pool[n][lab]

        if len(arr) < 2:
            return False

        j = rng.integers(0, len(arr))

        s = arr[j]

        test_sel[lab].append((n, s))

        pool[n][lab] = np.delete(arr, j)

        return True

    for lab in keep_labels:

        need = target[lab]

        names = list(counts.index)

        rng.shuffle(names)

        for n in names:

            if need <= 0:
                break

            if take_one(n, lab):
                need -= 1

    test_pairs = pd.DataFrame(
        [
            (n, s)
            for lab in keep_labels
            for (n, s) in test_sel[lab]
        ],
        columns=[name_col, session_col],
    ).drop_duplicates()

    train_list = []

    for n in counts.index:
        for lab in keep_labels:
            for s in pool[n][lab]:
                train_list.append((n, s))

    train_pairs = pd.DataFrame(
        train_list,
        columns=[name_col, session_col],
    ).drop_duplicates()

    df_train = df2.merge(
        train_pairs,
        on=[name_col, session_col],
    )

    df_test = df2.merge(
        test_pairs,
        on=[name_col, session_col],
    )

    return df_train, df_test


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def _safe_slug(s, maxlen=140):

    s2 = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s))
    s2 = re.sub(r"_+", "_", s2).strip("_")

    if len(s2) <= maxlen:
        return s2

    h = hashlib.md5(s.encode()).hexdigest()[:10]

    return s2[: maxlen - 11] + "_" + h


def ensure_numeric_new_class(df, col="new_class"):

    if pd.api.types.is_numeric_dtype(df[col]):
        return df

    codes, _ = pd.factorize(df[col].astype(str))

    out = df.copy()

    out[col] = codes.astype("int64")

    return out


def evaluate_binary(y_true, y_pred, pos_label="preictal"):

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["bckgd", pos_label],
        zero_division=0,
    )

    return {
        "f1_macro": float(f1_macro),
        "precision_preictal": float(pr[1]),
        "recall_preictal": float(rc[1]),
        "f1_preictal": float(f1[1]),
        "precision_bckgd": float(pr[0]),
        "recall_bckgd": float(rc[0]),
        "f1_bckgd": float(f1[0]),
        "support_bckgd": int(sup[0]),
        "support_preictal": int(sup[1]),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

lobe = pd.read_pickle(
    "/storage5/ame.lia/TUSZ-BIDS1/wide_topk3.pkl"
)

lobe = lobe[
    ~lobe["events_array"].isin(
        ["interictal", "postictal", "sz", "close_preictal"]
    )
].copy()


lobe, _ = zscore_by_subject_using_baseline(lobe)


labels = pd.read_csv(
    "/storage5/ame.lia/TUSZ-BIDS1/labels_ml.csv"
)

labels["name"] = labels.name.apply(
    lambda x: f"00{x}" if x < 10 else f"0{x}"
)

labels.set_index("name", inplace=True)

lobe = lobe[
    lobe["name"].isin(labels.index)
].copy()

lobe["new_class"] = lobe["name"].map(
    labels["label_k"]
)

lobe["id"] = (
    lobe["name"].astype(str)
    + ":"
    + lobe["session_run"].astype(str)
)

lobe = ensure_numeric_new_class(lobe)


selected = [
    c for c in lobe.columns
    if any(k in c for k in [
        "permen", "sampeen", "lz_complexity",
        "aproxen", "dfa", "exponent",
        "offset", "variance", "mean", "skewness"
    ])
]

W = 30

lobe = add_temporal_volatility_selective(
    lobe,
    window=W,
    min_periods=20,
    include_columns=selected,
)

lobe.dropna(inplace=True)


lobe = lobe[
    lobe["events_array"].isin(["bckgd", "preictal"])
].copy()


info_cols = [
    "events_array",
    "e_label",
    "name",
    "session_run",
    "time",
    "id",
]

base_features = [
    c for c in lobe.columns
    if c not in info_cols
    and "top1" in c
    and "vol30_mean" in c
]


df_train, df_test = split_sessions_within_name(
    lobe,
    test_size=0.2,
    seed=34,
    require_both_labels=False,
)


groups = df_train["id"].values

n_groups = len(np.unique(groups))

n_splits = max(2, min(5, n_groups))

cv = GroupKFold(n_splits=n_splits)


base_est = SVC(class_weight="balanced")

param_grid = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": [1e-3, 1e-2, 1e-1, "scale"],
}


OUTDIR = f"svm_single_W{W}"

os.makedirs(OUTDIR, exist_ok=True)

for d in ["models", "reports", "cv_results"]:
    os.makedirs(os.path.join(OUTDIR, d), exist_ok=True)


summary = []


for feat in sorted(base_features):

    x_cols = [feat, "new_class"]

    if df_train[feat].nunique() < 2:
        continue

    X_train = df_train[x_cols].values
    y_train = df_train["events_array"].values

    X_test = df_test[x_cols].values
    y_test = df_test["events_array"].values

    if not np.isfinite(X_train).all():
        continue

    gs = GridSearchCV(
        base_est,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    gs.fit(X_train, y_train, groups=groups)

    best = gs.best_estimator_

    y_pred = best.predict(X_test)

    metrics = evaluate_binary(y_test, y_pred)

    slug = _safe_slug(feat)

    model_path = os.path.join(
        OUTDIR,
        "models",
        f"svm__{slug}.joblib",
    )

    joblib.dump(best, model_path)

    summary.append({
        "feature": feat,
        "cv_f1": gs.best_score_,
        "test_f1": metrics["f1_macro"],
        "test_f1_preictal": metrics["f1_preictal"],
        "model_path": model_path,
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(
        OUTDIR,
        "summary_single_feature_models.csv"
    ),
    index=False,
)

print("Done.")
