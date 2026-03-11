"""
validation.privacy_metrics
===========================

Privacy-specific metrics for evaluating anonymised or synthetic tabular data.

Each function produces a scalar (or small summary dict) that quantifies a
distinct privacy risk.  All functions are dataset-agnostic; column names are
passed explicitly via parameters so that no assumptions are made about the
schema of the underlying data.

Typical usage
-------------
>>> from validation.privacy_metrics import membership_inference_auc
>>> auc = membership_inference_auc(real_df, synth_df, numerical_cols=["age", "income"])
>>> # Closer to 0.5 means better privacy protection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Membership inference
# ---------------------------------------------------------------------------


def membership_inference_auc(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols: list[str],
    seed: int = 42,
) -> float:
    """Estimate privacy risk via a membership-inference classifier.

    Trains a logistic regression binary classifier to distinguish real rows
    (label 1) from synthetic rows (label 0).  Features are built from
    ``numerical_cols`` (standardised) plus any remaining shared columns that
    are one-hot encoded.  The dataset is split into train/test sets and the
    resulting ROC-AUC on the held-out split is returned.

    Interpretation:
        - AUC ≈ 0.5  the classifier cannot distinguish real from synthetic
          (good privacy).
        - AUC ≈ 1.0  the synthetic data is easily distinguishable from real
          (poor privacy, high re-identification risk).

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        numerical_cols: Column names to treat as numerical features.
            These are standardised (zero-mean, unit-variance) before training.
            All remaining shared columns are treated as categorical and are
            one-hot encoded.
        seed: Random seed for the train/test split and classifier.

    Returns:
        ROC-AUC score in [0.0, 1.0].  Returns 0.5 on error or when there are
        no shared columns to work with.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        logger.error("scikit-learn is required for membership_inference_auc: %s", exc)
        return 0.5

    shared_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not shared_cols:
        logger.warning(
            "No shared columns between real and synthetic DataFrames; returning 0.5."
        )
        return 0.5

    real_sub = real_df[shared_cols].copy()
    synth_sub = synth_df[shared_cols].copy()
    real_sub["__label__"] = 1
    synth_sub["__label__"] = 0

    combined = pd.concat([real_sub, synth_sub], ignore_index=True)
    X = combined[shared_cols]
    y = combined["__label__"].values

    # Resolve numerical / categorical feature split.
    num_feats = [c for c in numerical_cols if c in shared_cols]
    cat_feats = [
        c
        for c in shared_cols
        if c not in num_feats
        and (
            X[c].dtype == object
            or X[c].dtype.name == "category"
            or pd.api.types.is_string_dtype(X[c])
        )
    ]
    # Any remaining column not yet classified is treated as numeric.
    classified = set(num_feats) | set(cat_feats)
    num_feats = num_feats + [c for c in shared_cols if c not in classified]

    transformers: list = []
    if num_feats:
        transformers.append(("num", StandardScaler(), num_feats))
    if cat_feats:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_feats,
            )
        )

    if not transformers:
        logger.warning("No usable feature columns; returning 0.5.")
        return 0.5

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = Pipeline(
        [
            ("pre", preprocessor),
            ("lr", LogisticRegression(max_iter=500, random_state=seed)),
        ]
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception as exc:
        logger.error("Membership inference AUC failed: %s", exc)
        return 0.5

    logger.info("Membership inference AUC: %.4f  (0.5 = best privacy)", auc)
    return auc


# ---------------------------------------------------------------------------
# Duplicate / novel row rates
# ---------------------------------------------------------------------------


def duplicate_class_rate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> float:
    """Compute the fraction of synthetic rows that exactly match a real row.

    Exact matching is performed on ``key_columns`` (or all shared columns when
    ``key_columns`` is ``None``).  Both DataFrames are coerced to string before
    comparison so that mixed-type columns are handled consistently.

    A rate close to 0.0 indicates good privacy (no synthetic row is a
    verbatim copy of a real record).  A rate close to 1.0 is a red flag.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        key_columns: Subset of columns to use for comparison.  When
            ``None``, all columns present in both DataFrames are used.

    Returns:
        Fraction in [0.0, 1.0].  Returns 0.0 if ``synth_df`` is empty or
        no shared columns are available.
    """
    if synth_df.empty:
        return 0.0

    if key_columns is not None:
        shared_cols = [
            c for c in key_columns if c in real_df.columns and c in synth_df.columns
        ]
    else:
        shared_cols = [c for c in synth_df.columns if c in real_df.columns]

    if not shared_cols:
        logger.warning("No shared columns for duplicate_class_rate; returning 0.0.")
        return 0.0

    real_set = set(real_df[shared_cols].astype(str).apply(tuple, axis=1))
    synth_tuples = synth_df[shared_cols].astype(str).apply(tuple, axis=1)

    duplicate_count = int(synth_tuples.isin(real_set).sum())
    rate = duplicate_count / len(synth_df)

    logger.info(
        "Duplicate class rate: %.4f  (%d / %d synthetic rows are exact real copies).",
        rate,
        duplicate_count,
        len(synth_df),
    )
    return float(rate)


def new_row_rate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> float:
    """Compute the fraction of synthetic rows that are entirely novel.

    A synthetic row is considered novel if it does not exactly match any row
    in the real DataFrame on ``key_columns`` (or all shared columns when
    ``key_columns`` is ``None``).

    A rate close to 1.0 is ideal: every synthetic record is genuinely new
    and cannot be trivially linked back to a real individual.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        key_columns: Subset of columns to use for comparison.  When
            ``None``, all columns present in both DataFrames are used.

    Returns:
        Fraction in [0.0, 1.0].  Equal to ``1 - duplicate_class_rate``.
    """
    dup_rate = duplicate_class_rate(real_df, synth_df, key_columns=key_columns)
    novel_rate = 1.0 - dup_rate
    logger.info("New row rate: %.4f", novel_rate)
    return novel_rate


# ---------------------------------------------------------------------------
# Nearest-neighbour distance
# ---------------------------------------------------------------------------


def nearest_neighbor_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols: list[str],
    n_samples: int = 1_000,
    seed: int = 42,
) -> dict[str, float]:
    """Summarise the distribution of nearest-neighbour distances (synth to real).

    A random sample of up to ``n_samples`` synthetic rows is drawn.  For each
    sampled row the Euclidean distance to the nearest real row (in the
    standardised numerical feature space) is computed using
    ``sklearn.neighbors.NearestNeighbors``.

    Higher distances indicate that synthetic records are less similar to any
    individual real record, implying better privacy protection.  A
    distribution heavily concentrated near zero suggests near-verbatim copies
    of real rows.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        numerical_cols: Numerical column names used for the distance
            computation.  Columns absent from either DataFrame are dropped.
        n_samples: Maximum number of synthetic rows to sample.  Sampling is
            deterministic given ``seed``.
        seed: Random seed for reproducible sampling.

    Returns:
        Dict with keys ``mean_distance``, ``median_distance``,
        ``min_distance``, ``p5_distance``, and ``p25_distance``.  All values
        are ``float``.  Returns a dict of ``NaN`` values when no suitable
        columns are available or either DataFrame is empty.
    """
    _nan_result: dict[str, float] = {
        "mean_distance": float("nan"),
        "median_distance": float("nan"),
        "min_distance": float("nan"),
        "p5_distance": float("nan"),
        "p25_distance": float("nan"),
    }

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        logger.error("scikit-learn is required for nearest_neighbor_distance: %s", exc)
        return _nan_result

    present_cols = [
        c for c in numerical_cols if c in real_df.columns and c in synth_df.columns
    ]
    if not present_cols:
        logger.warning(
            "No numerical columns available for nearest_neighbor_distance."
        )
        return _nan_result

    real_vals = real_df[present_cols].fillna(0.0).values.astype(float)
    synth_vals = synth_df[present_cols].fillna(0.0).values.astype(float)

    if real_vals.shape[0] == 0 or synth_vals.shape[0] == 0:
        logger.warning(
            "One DataFrame is empty; cannot compute nearest-neighbour distances."
        )
        return _nan_result

    # Standardise on real statistics so both sets are in the same feature space.
    col_means = real_vals.mean(axis=0)
    col_stds = real_vals.std(axis=0)
    col_stds[col_stds == 0.0] = 1.0  # guard against constant columns

    real_norm = (real_vals - col_means) / col_stds

    # Subsample synthetic rows for scalability.
    rng = np.random.default_rng(seed)
    n_synth = synth_vals.shape[0]
    if n_synth > n_samples:
        idx = rng.choice(n_synth, size=n_samples, replace=False)
        synth_sample = synth_vals[idx]
    else:
        synth_sample = synth_vals

    synth_norm = (synth_sample - col_means) / col_stds

    try:
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nn.fit(real_norm)
        distances, _ = nn.kneighbors(synth_norm)
        nn_distances = distances.flatten()
    except Exception as exc:
        logger.error("Nearest-neighbour distance computation failed: %s", exc)
        return _nan_result

    result: dict[str, float] = {
        "mean_distance": float(np.mean(nn_distances)),
        "median_distance": float(np.median(nn_distances)),
        "min_distance": float(np.min(nn_distances)),
        "p5_distance": float(np.percentile(nn_distances, 5)),
        "p25_distance": float(np.percentile(nn_distances, 25)),
    }

    logger.info(
        "NN distances : mean: %.4f  median: %.4f  min: %.4f  p5: %.4f  p25: %.4f",
        result["mean_distance"],
        result["median_distance"],
        result["min_distance"],
        result["p5_distance"],
        result["p25_distance"],
    )
    return result
