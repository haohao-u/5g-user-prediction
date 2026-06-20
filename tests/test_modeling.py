import pandas as pd

from src.modeling import (
    CAT_FEATURES,
    NUM_FEATURES,
    TARGET,
    build_preprocessor,
    create_models,
    evaluate_models,
    split_features_target,
)


def sample_frame():
    data = {
        "id": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
    }
    for name in CAT_FEATURES:
        data[name] = [0, 1, 0, 1]
    for name in NUM_FEATURES:
        data[name] = [0.1, 0.2, 0.3, 0.4]
    return pd.DataFrame(data)


def test_feature_schema_matches_assignment():
    assert len(CAT_FEATURES) == 20
    assert len(NUM_FEATURES) == 38
    assert TARGET == "target"


def test_split_features_target_removes_id_and_target():
    x, y = split_features_target(sample_frame())

    assert "id" not in x.columns
    assert TARGET not in x.columns
    assert y.tolist() == [0, 1, 0, 1]


def test_preprocessor_transforms_expected_columns():
    x, _ = split_features_target(sample_frame())
    transformed = build_preprocessor().fit_transform(x)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] == len(CAT_FEATURES) + len(NUM_FEATURES)


def test_models_and_evaluation_return_auc_scores():
    x, y = split_features_target(sample_frame())
    models = create_models(random_state=7)

    assert set(models) == {"logistic_regression", "random_forest"}

    scores = evaluate_models(models, x, y, x, y)

    assert set(scores) == {"logistic_regression", "random_forest"}
    assert all(0.0 <= score <= 1.0 for score in scores.values())
