"""
End-to-end training pipeline for cyberbullying detection.
Implements:
1) Reproducible run with saved artifacts.
2) Required evaluation outputs.
3) Train vs test comparison for overfitting checks.
4) Hyperparameter tuning for two models.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.text_preprocessing import TextPreprocessor
from src.vectorization import vectorize_texts


SEED = 42
DATA_PATH = os.path.join("data", "raw", "cyberbullying_tweets.csv")
RESULTS_DIR = "results"


@dataclass
class ModelBundle:
    name: str
    model: object


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def tune_models(X_train, y_train) -> Tuple[ModelBundle, ModelBundle, pd.DataFrame]:
    # Keep grids compact to reduce training time while still doing proper selection.
    lr_grid = {
        "C": [0.5, 1.0, 2.0],
        "class_weight": [None, "balanced"],
    }
    gb_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }

    lr_base = LogisticRegression(max_iter=1000, random_state=SEED)
    gb_base = GradientBoostingClassifier(random_state=SEED)

    lr_search = GridSearchCV(
        estimator=lr_base,
        param_grid=lr_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
    )
    gb_search = GridSearchCV(
        estimator=gb_base,
        param_grid=gb_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
    )

    lr_search.fit(X_train, y_train)
    gb_search.fit(X_train, y_train)

    tuning_rows = [
        {
            "model": "Logistic Regression",
            "best_score_cv_f1_weighted": lr_search.best_score_,
            "best_params": json.dumps(lr_search.best_params_, ensure_ascii=True),
        },
        {
            "model": "Gradient Boosting",
            "best_score_cv_f1_weighted": gb_search.best_score_,
            "best_params": json.dumps(gb_search.best_params_, ensure_ascii=True),
        },
    ]

    return (
        ModelBundle("Logistic Regression (tuned)", lr_search.best_estimator_),
        ModelBundle("Gradient Boosting (tuned)", gb_search.best_estimator_),
        pd.DataFrame(tuning_rows),
    )


def evaluate_models(models: List[ModelBundle], X_train, y_train, X_test, y_test) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    test_rows: List[Dict[str, float]] = []
    train_test_rows: List[Dict[str, float]] = []
    preds_test: Dict[str, np.ndarray] = {}
    preds_train: Dict[str, np.ndarray] = {}

    for model_bundle in models:
        model = model_bundle.model
        name = model_bundle.name

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        preds_train[name] = y_pred_train
        preds_test[name] = y_pred_test

        train_m = weighted_metrics(y_train, y_pred_train)
        test_m = weighted_metrics(y_test, y_pred_test)

        test_rows.append({"Model": name, **test_m})

        for metric in ["accuracy", "precision", "recall", "f1"]:
            train_test_rows.append(
                {
                    "Model": name,
                    "Metric": metric,
                    "Train": train_m[metric],
                    "Test": test_m[metric],
                    "Gap": train_m[metric] - test_m[metric],
                }
            )

    return pd.DataFrame(test_rows), pd.DataFrame(train_test_rows), preds_test, preds_train


def save_model_comparison_plot(metrics_df: pd.DataFrame) -> None:
    metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metrics_long, x="Metric", y="Value", hue="Model")
    plt.ylim(0, 1)
    plt.title("Model Comparison (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=300)
    plt.close()


def save_train_test_gap_plot(train_test_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=train_test_df, x="Metric", y="Gap", hue="Model")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Train-Test Metric Gap (Train - Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "train_vs_test_gap.png"), dpi=300)
    plt.close()


def save_best_model_artifacts(
    best_model_name: str,
    y_test: np.ndarray,
    y_pred_best: np.ndarray,
    label_names: List[str],
    X_test_text: pd.Series,
) -> None:
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_best.png"), dpi=300)
    plt.close()

    report = classification_report(y_test, y_pred_best, target_names=label_names, digits=4)
    with open(os.path.join(RESULTS_DIR, "classification_report_best.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    ok_idx = np.where(y_pred_best == y_test)[0][:5]
    fail_idx = np.where(y_pred_best != y_test)[0][:5]

    with open(os.path.join(RESULTS_DIR, "examples_correct_incorrect.txt"), "w", encoding="utf-8") as f:
        f.write("Examples correctly classified:\n")
        for idx in ok_idx:
            f.write(f"- {X_test_text.iloc[idx][:160]}\n")

        f.write("\nExamples misclassified:\n")
        for idx in fail_idx:
            f.write(f"- {X_test_text.iloc[idx][:160]} | pred={y_pred_best[idx]} true={y_test[idx]}\n")


def save_user_test_predictions(preprocessor: TextPreprocessor, vectorizer, model, inv_label_map: Dict[int, str]) -> None:
    samples = [
        "I hope you have a great day!",
        "You are stupid and nobody wants you",
        "Go back to your country, we do not want you here",
        "I disagree with your opinion but respect your right to express it",
    ]

    lines: List[str] = []
    for sample in samples:
        cleaned = preprocessor.process_text(sample)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        lines.append(f"Tweet: {sample}")
        lines.append(f"Cleaned: {cleaned}")
        lines.append(f"Prediction: {inv_label_map[int(pred)]}")
        lines.append("-" * 80)

    with open(os.path.join(RESULTS_DIR, "user_test_predictions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    started = time.time()

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["tweet_text", "cyberbullying_type"]).copy()

    pre = TextPreprocessor()
    df["tweet_text_cleaned"] = df["tweet_text"].apply(pre.process_text)

    label_map = {label: idx for idx, label in enumerate(sorted(df["cyberbullying_type"].unique()))}
    inv_label_map = {v: k for k, v in label_map.items()}

    y = df["cyberbullying_type"].map(label_map).to_numpy()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["tweet_text_cleaned"],
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    X_train, X_test, tfidf_vectorizer = vectorize_texts(
        X_train_text,
        X_test_text,
        vectorizer_type="tfidf",
        max_features=5000,
    )

    nb_bundle = ModelBundle("Naive Bayes", MultinomialNB(alpha=0.01))
    lr_tuned_bundle, gb_tuned_bundle, tuning_df = tune_models(X_train, y_train)

    models = [nb_bundle, lr_tuned_bundle, gb_tuned_bundle]
    metrics_df, train_test_df, preds_test, _ = evaluate_models(models, X_train, y_train, X_test, y_test)

    metrics_df = metrics_df[["Model", "accuracy", "precision", "recall", "f1"]].sort_values("f1", ascending=False)
    train_test_df = train_test_df.sort_values(["Metric", "Model"]).reset_index(drop=True)

    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"), index=False)
    train_test_df.to_csv(os.path.join(RESULTS_DIR, "train_vs_test_comparison.csv"), index=False)
    tuning_df.to_csv(os.path.join(RESULTS_DIR, "hyperparameter_tuning.csv"), index=False)

    save_model_comparison_plot(metrics_df)
    save_train_test_gap_plot(train_test_df)

    best_name = metrics_df.iloc[0]["Model"]
    y_pred_best = preds_test[best_name]

    save_best_model_artifacts(
        best_model_name=best_name,
        y_test=y_test,
        y_pred_best=y_pred_best,
        label_names=[inv_label_map[i] for i in range(len(inv_label_map))],
        X_test_text=X_test_text.reset_index(drop=True),
    )

    name_to_model = {bundle.name: bundle.model for bundle in models}
    save_user_test_predictions(pre, tfidf_vectorizer, name_to_model[best_name], inv_label_map)

    run_manifest = {
        "seed": SEED,
        "dataset_path": DATA_PATH,
        "n_samples": int(df.shape[0]),
        "n_classes": int(len(label_map)),
        "models_evaluated": [bundle.name for bundle in models],
        "best_model": best_name,
        "elapsed_seconds": round(time.time() - started, 2),
    }

    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("Run completed successfully.")
    print(f"Best model: {best_name}")
    print(f"Artifacts saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
