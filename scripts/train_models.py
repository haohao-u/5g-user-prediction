from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution with `python scripts/train_models.py` from the project root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from src.modeling import (
    CAT_FEATURES,
    NUM_FEATURES,
    TARGET,
    create_models,
    load_training_data,
    split_features_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 5G user prediction models.")
    parser.add_argument("--data", default="train.csv", help="Path to train.csv")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100000,
        help="Rows to load for a quick reproducible run. Use 0 for all rows.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def save_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the class balance chart used in the coursework analysis."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET, hue=TARGET, palette=["#496A81", "#D17A22"], legend=False)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Whether the user is a 5G user")
    ax.set_ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close()


def save_auc_plot(scores: dict[str, float], output_dir: Path) -> None:
    """Save a compact comparison chart for validation AUC scores."""
    score_frame = pd.DataFrame(
        {"model": list(scores.keys()), "auc": list(scores.values())}
    ).sort_values("auc", ascending=False)
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=score_frame, x="auc", y="model", hue="model", palette="Set2", legend=False)
    ax.set_xlim(0, 1)
    ax.set_title("Validation AUC Comparison")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Model")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f")
    plt.tight_layout()
    plt.savefig(output_dir / "auc_comparison.png", dpi=160)
    plt.close()


def save_roc_curves(models, x_valid, y_valid, output_dir: Path) -> None:
    """Save ROC curves for the trained validation models."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, x_valid, y_valid, name=name, ax=ax)
    ax.set_title("Validation ROC Curves")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_rows = None if args.max_rows == 0 else args.max_rows
    df = load_training_data(args.data, max_rows=max_rows)
    x, y = split_features_target(df)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        # Stratification keeps the rare positive class ratio stable in both splits.
        stratify=y,
    )

    models = create_models(random_state=args.random_state)
    scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        from sklearn.metrics import roc_auc_score

        scores[name] = float(roc_auc_score(y_valid, probabilities))

    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    # Keep machine-readable metrics so results can be reproduced or reported later.
    summary = {
        "rows_used": int(len(df)),
        "positive_rate": float(y.mean()),
        "feature_count": len(CAT_FEATURES) + len(NUM_FEATURES),
        "categorical_features": len(CAT_FEATURES),
        "numeric_features": len(NUM_FEATURES),
        "scores": scores,
        "best_model": best_model_name,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([{"model": name, "auc": auc} for name, auc in scores.items()]).to_csv(
        output_dir / "model_scores.csv",
        index=False,
    )
    joblib.dump(best_model, output_dir / "best_model.joblib")
    save_target_distribution(df, output_dir)
    save_auc_plot(scores, output_dir)
    save_roc_curves(models, x_valid, y_valid, output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
