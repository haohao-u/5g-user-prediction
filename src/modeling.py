from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


TARGET = "target"
ID_COLUMN = "id"
CAT_FEATURES = [f"cat_{i}" for i in range(20)]
NUM_FEATURES = [f"num_{i}" for i in range(38)]
FEATURES = CAT_FEATURES + NUM_FEATURES


def load_training_data(csv_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load the assignment training CSV with an optional row cap for quick runs."""
    return pd.read_csv(csv_path, nrows=max_rows)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in [TARGET, *FEATURES] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return x, y


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical, CAT_FEATURES),
            ("num", numeric, NUM_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Create the two required AI algorithms as reproducible sklearn pipelines."""
    logistic_regression = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def evaluate_models(
    models: Mapping[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        scores[name] = float(roc_auc_score(y_valid, probabilities))
    return scores
