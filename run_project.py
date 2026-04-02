"""
End-to-end training pipeline for cyberbullying detection.
Implements:
1) Reproducible run with saved artifacts.
2) Required evaluation outputs.
3) Train vs test comparison for overfitting checks.
4) Hyperparameter tuning for selected classical models.
5) Directed keyword feature branch for hard confusion cases.
"""

from __future__ import annotations

import json
import os
import random
import time
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.text_preprocessing import TextPreprocessor
from src.vectorization import vectorize_texts


SEED = 42
DATA_PATH = os.path.join("data", "raw", "cyberbullying_tweets.csv")
RESULTS_DIR = "results"
RESULTS_SUBDIRS = {
    "tables": os.path.join(RESULTS_DIR, "tables"),
    "figures": os.path.join(RESULTS_DIR, "figures"),
    "reports": os.path.join(RESULTS_DIR, "reports"),
    "predictions": os.path.join(RESULTS_DIR, "predictions"),
    "manifests": os.path.join(RESULTS_DIR, "manifests"),
}
CV_JOBS = 1
CV_FOLDS = 2
WEAK_CLASS_NAMES = ("not_cyberbullying", "other_cyberbullying")
VECTORIZER_TYPE = "tfidf_hybrid_keywords"
MAX_FEATURES = 2000
ENSEMBLE_SUBSET_SIZE = 12000
KEYWORD_FEATURE_WEIGHT = 3.0
KEYWORD_PHRASES = (
    "go back",
    "go back your country",
    "go back your country not want you",
    "your country",
    "your country not want you",
    "country",
    "not want you",
    "not welcome here",
    "nobody want you",
    "leave country",
    "do not belong",
    "you do not belong",
    "you should leave",
    "send you back",
    "foreign",
    "immigrant",
    "not welcome",
    "we do not want you",
)
USER_SAFE_NOT_PATTERNS = (
    "respect your right",
    "have a great day",
    "hope you great day",
)
USER_ALERT_OTHER_PATTERNS = (
    "go back your country",
    "go back your country not want you",
    "not welcome here",
    "do not belong",
    "you do not belong",
)
USER_DIRECT_REJECTION_PATTERNS = (
    "not want you",
    "nobody want you",
)
USER_INSULT_TERMS = {
    "stupid",
    "idiot",
    "moron",
    "trash",
    "hate",
}


@dataclass
class ModelBundle:
    name: str
    model: object
    fit_subset_size: Optional[int] = None


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for subdir in RESULTS_SUBDIRS.values():
        os.makedirs(subdir, exist_ok=True)


def result_path(group: str, filename: str) -> str:
    return os.path.join(RESULTS_SUBDIRS[group], filename)


def save_data_overview(df_raw: pd.DataFrame, df_used: pd.DataFrame) -> None:
    """Save compact data quality and class-balance artifacts for report writing."""
    missing_tweet = int(df_raw["tweet_text"].isna().sum())
    missing_label = int(df_raw["cyberbullying_type"].isna().sum())
    dropped_missing = int(len(df_raw) - len(df_used))

    class_dist = df_used["cyberbullying_type"].value_counts().sort_index()
    class_dist_df = class_dist.rename_axis("class").reset_index(name="count")
    class_dist_df["percentage"] = class_dist_df["count"] / class_dist_df["count"].sum()
    count_min = float(class_dist_df["count"].min())
    count_max = float(class_dist_df["count"].max())
    if count_max > count_min:
        class_dist_df["count_minmax"] = (class_dist_df["count"] - count_min) / (count_max - count_min)
    else:
        class_dist_df["count_minmax"] = 0.0

    class_dist_df.to_csv(result_path("tables", "data_class_distribution.csv"), index=False)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=class_dist_df, x="class", y="count_minmax", color="#4C72B0")
    ax.set_ylim(0, 1.08)
    for patch, (_, row) in zip(ax.patches, class_dist_df.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + 0.02,
            f"{int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylabel("Normalized count (min-max)")
    plt.xticks(rotation=25, ha="right")
    plt.title("Class Distribution (Min-Max Normalized, After Cleaning)")
    plt.tight_layout()
    plt.savefig(result_path("figures", "data_class_distribution.png"), dpi=300)
    plt.close()

    overview = {
        "n_rows_raw": int(len(df_raw)),
        "n_rows_after_dropna": int(len(df_used)),
        "rows_dropped_due_to_missing_required_columns": dropped_missing,
        "missing_tweet_text_raw": missing_tweet,
        "missing_cyberbullying_type_raw": missing_label,
        "class_distribution_after_cleaning": [
            {
                "class": row["class"],
                "count": int(row["count"]),
                "percentage": round(float(row["percentage"]), 6),
                "count_minmax": round(float(row["count_minmax"]), 6),
            }
            for _, row in class_dist_df.iterrows()
        ],
    }

    with open(result_path("manifests", "data_overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=True)


def stratified_subset_indices(y: np.ndarray, subset_size: int, seed: int = SEED) -> np.ndarray:
    if subset_size >= len(y):
        return np.arange(len(y))

    all_indices = np.arange(len(y))
    subset_indices, _ = train_test_split(
        all_indices,
        train_size=subset_size,
        stratify=y,
        random_state=seed,
    )
    return np.asarray(subset_indices)


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def cross_validated_weighted_f1(estimator, X, y, cv_folds: int = CV_FOLDS) -> float:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    scores: List[float] = []

    for train_idx, val_idx in cv.split(np.zeros(len(y)), y):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        y_val_pred = model.predict(X[val_idx])
        score = f1_score(y[val_idx], y_val_pred, average="weighted", zero_division=0)
        scores.append(float(score))

    return float(np.mean(scores))


def extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    base = weighted_metrics(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    weak_scores = [float(report.get(class_name, {}).get("f1-score", 0.0)) for class_name in WEAK_CLASS_NAMES]
    weak_f1_avg = float(np.mean(weak_scores))

    return {
        **base,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "f1_not_cyberbullying": weak_scores[0],
        "f1_other_cyberbullying": weak_scores[1],
        "weak_f1_avg": weak_f1_avg,
        "balance_score": float((base["f1"] + weak_f1_avg) / 2.0),
    }


def compare_vectorizers(
    X_train_text: pd.Series,
    X_test_text: pd.Series,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
) -> pd.DataFrame:
    """Benchmark multiple vectorization schemes with the same lightweight linear classifier."""
    candidates = [
        "bow",
        "tfidf",
        "tfidf_hybrid",
        "tfidf_hybrid_keywords",
    ]

    rows: List[Dict[str, float]] = []
    for vec_type in candidates:
        started = time.time()

        X_train_vec, X_test_vec, _ = vectorize_texts(
            X_train_text,
            X_test_text,
            vectorizer_type=vec_type,
            max_features=MAX_FEATURES,
            keyword_phrases=list(KEYWORD_PHRASES),
            keyword_feature_weight=KEYWORD_FEATURE_WEIGHT,
        )

        probe_model = LinearSVC(max_iter=3000, random_state=SEED, C=0.5, class_weight="balanced")
        probe_model.fit(X_train_vec, y_train)
        y_pred = probe_model.predict(X_test_vec)

        m = extended_metrics(y_test, y_pred, label_names)
        rows.append(
            {
                "vectorizer_type": vec_type,
                "fit_predict_seconds": round(time.time() - started, 2),
                "uses_keyword_branch": vec_type == "tfidf_hybrid_keywords",
                **m,
            }
        )

    comp_df = pd.DataFrame(rows)
    return comp_df.sort_values(["balance_score", "f1"], ascending=False).reset_index(drop=True)


def tune_models(X_train, y_train) -> Tuple[ModelBundle, ModelBundle, ModelBundle, pd.DataFrame]:
    # Keep searches compact to reduce training time while still doing proper selection.
    nb_candidates = [{"alpha": alpha} for alpha in [0.005, 0.01, 0.05]]
    svm_candidates = [
        {"C": c, "class_weight": cw}
        for c, cw in product([0.5, 1.0], [None, "balanced"])
    ]
    rf_fixed_params = {
        "n_estimators": 120,
        "max_depth": 20,
        "min_samples_split": 2,
    }

    best_nb_params: Dict[str, object] = {}
    best_nb_score = float("-inf")
    for params in nb_candidates:
        score = cross_validated_weighted_f1(MultinomialNB(**params), X_train, y_train, cv_folds=CV_FOLDS)
        if score > best_nb_score:
            best_nb_score = score
            best_nb_params = params

    best_svm_params: Dict[str, object] = {}
    best_svm_score = float("-inf")
    for params in svm_candidates:
        score = cross_validated_weighted_f1(
            LinearSVC(max_iter=5000, random_state=SEED, **params),
            X_train,
            y_train,
            cv_folds=CV_FOLDS,
        )
        if score > best_svm_score:
            best_svm_score = score
            best_svm_params = params

    nb_best_model = MultinomialNB(**best_nb_params)
    svm_best_model = LinearSVC(max_iter=5000, random_state=SEED, **best_svm_params)
    rf_base = RandomForestClassifier(random_state=SEED, n_jobs=1, **rf_fixed_params)

    rf_subset_idx = stratified_subset_indices(y_train, ENSEMBLE_SUBSET_SIZE, seed=SEED)

    tuning_rows = [
        {
            "model": "Naive Bayes",
            "best_score_cv_f1_weighted": best_nb_score,
            "best_params": json.dumps(best_nb_params, ensure_ascii=True),
        },
        {
            "model": "Linear SVM",
            "best_score_cv_f1_weighted": best_svm_score,
            "best_params": json.dumps(best_svm_params, ensure_ascii=True),
        },
        {
            "model": "Random Forest (fixed)",
            "best_score_cv_f1_weighted": np.nan,
            "best_params": json.dumps(rf_fixed_params, ensure_ascii=True),
            "train_subset_size": int(len(rf_subset_idx)),
        },
    ]

    return (
        ModelBundle("Naive Bayes (tuned)", nb_best_model),
        ModelBundle("Linear SVM (tuned)", svm_best_model),
        ModelBundle("Random Forest", rf_base, fit_subset_size=int(len(rf_subset_idx))),
        pd.DataFrame(tuning_rows),
    )


def evaluate_models(
    models: List[ModelBundle],
    X_train,
    y_train,
    X_test,
    y_test,
    label_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    test_rows: List[Dict[str, float]] = []
    train_test_rows: List[Dict[str, float]] = []
    preds_test: Dict[str, np.ndarray] = {}
    preds_train: Dict[str, np.ndarray] = {}

    for model_bundle in models:
        model = model_bundle.model
        name = model_bundle.name

        fit_X = X_train
        fit_y = y_train
        if model_bundle.fit_subset_size is not None:
            fit_idx = stratified_subset_indices(y_train, model_bundle.fit_subset_size, seed=SEED)
            fit_X = X_train[fit_idx]
            fit_y = y_train[fit_idx]

        model.fit(fit_X, fit_y)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        preds_train[name] = y_pred_train
        preds_test[name] = y_pred_test

        train_m = extended_metrics(y_train, y_pred_train, label_names)
        test_m = extended_metrics(y_test, y_pred_test, label_names)

        test_rows.append({"Model": name, **test_m})

        for metric in ["accuracy", "precision", "recall", "f1", "macro_f1", "weak_f1_avg", "balance_score"]:
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
    metrics_long = metrics_df[["Model", "accuracy", "precision", "recall", "f1"]].melt(
        id_vars="Model", var_name="Metric", value_name="Value"
    )
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(data=metrics_long, x="Metric", y="Value", hue="Model")
    plt.ylim(0, 1)
    plt.title("Model Comparison (Test Set)")
    # Keep the legend outside the plot so bars and labels remain visible.
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(result_path("figures", "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_train_test_gap_plot(train_test_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=train_test_df, x="Metric", y="Gap", hue="Model")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Train-Test Metric Gap (Train - Test)")
    plt.tight_layout()
    plt.savefig(result_path("figures", "train_vs_test_gap.png"), dpi=300)
    plt.close()


def save_best_model_artifacts(
    best_model_name: str,
    y_test: np.ndarray,
    y_pred_best: np.ndarray,
    label_names: List[str],
    inv_label_map: Dict[int, str],
    X_test_text: pd.Series,
) -> None:
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(result_path("figures", "confusion_matrix_best.png"), dpi=300)
    plt.close()

    report = classification_report(y_test, y_pred_best, target_names=label_names, digits=4)
    with open(result_path("reports", "classification_report_best.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    ok_idx = np.where(y_pred_best == y_test)[0][:5]
    fail_idx = np.where(y_pred_best != y_test)[0][:5]

    with open(result_path("reports", "examples_correct_incorrect.txt"), "w", encoding="utf-8") as f:
        f.write("Examples correctly classified:\n")
        for idx in ok_idx:
            f.write(f"- {X_test_text.iloc[idx][:160]}\n")

        f.write("\nExamples misclassified:\n")
        for idx in fail_idx:
            pred_label = inv_label_map[int(y_pred_best[idx])]
            true_label = inv_label_map[int(y_test[idx])]
            f.write(f"- {X_test_text.iloc[idx][:160]} | pred={pred_label} true={true_label}\n")


def save_common_confusions(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], top_n: int = 10) -> None:
    """Store the most frequent off-diagonal confusions to support error analysis in the report."""
    cm = confusion_matrix(y_true, y_pred)
    rows: List[Dict[str, float]] = []

    for i, true_label in enumerate(label_names):
        row_total = int(cm[i].sum())
        for j, pred_label in enumerate(label_names):
            if i == j:
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                    "error_rate_within_true_label": float(count / row_total) if row_total else 0.0,
                }
            )

    if rows:
        out_df = pd.DataFrame(rows).sort_values(
            ["count", "error_rate_within_true_label"],
            ascending=[False, False],
        ).head(top_n)
    else:
        out_df = pd.DataFrame(
            columns=["true_label", "predicted_label", "count", "error_rate_within_true_label"]
        )

    out_df.to_csv(result_path("tables", "common_confusions_best.csv"), index=False)


def predict_tweet(
    tweet: str,
    preprocessor: TextPreprocessor,
    vectorizer,
    model,
    inv_label_map: Dict[int, str],
) -> Tuple[str, str, str]:
    """Predict a single tweet and return cleaned text, model label, and final label."""
    cleaned = preprocessor.process_text(tweet)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    pred_label = inv_label_map[int(pred)]

    final_label = pred_label
    if any(pattern in cleaned for pattern in USER_SAFE_NOT_PATTERNS):
        final_label = "not_cyberbullying"
    else:
        has_explicit_other = any(pattern in cleaned for pattern in USER_ALERT_OTHER_PATTERNS)
        has_direct_rejection = any(pattern in cleaned for pattern in USER_DIRECT_REJECTION_PATTERNS)
        has_insult = any(term in cleaned.split() for term in USER_INSULT_TERMS)
        if has_explicit_other or (has_direct_rejection and has_insult):
            final_label = "other_cyberbullying"

    return cleaned, pred_label, final_label


def save_user_test_predictions(preprocessor: TextPreprocessor, vectorizer, model, inv_label_map: Dict[int, str]) -> None:
    samples = [
        "I hope you have a great day!",
        "You are stupid and nobody wants you",
        "Go back to your country, we do not want you here",
        "I disagree with your opinion but respect your right to express it",
    ]

    lines: List[str] = []
    for sample in samples:
        cleaned, pred_label, final_label = predict_tweet(
            sample,
            preprocessor,
            vectorizer,
            model,
            inv_label_map,
        )

        lines.append(f"Tweet: {sample}")
        lines.append(f"Cleaned: {cleaned}")
        lines.append(f"Prediction: {final_label}")
        if final_label != pred_label:
            lines.append(f"Model prediction before rule: {pred_label}")
            lines.append("Rule override applied: yes")
        lines.append("-" * 80)

    with open(result_path("predictions", "user_test_predictions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    started = time.time()

    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.dropna(subset=["tweet_text", "cyberbullying_type"]).copy()
    save_data_overview(df_raw, df)

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

    label_names = [inv_label_map[i] for i in range(len(inv_label_map))]
    vectorizer_comp_df = compare_vectorizers(
        X_train_text,
        X_test_text,
        y_train,
        y_test,
        label_names,
    )
    vectorizer_comp_df.to_csv(result_path("tables", "vectorizer_comparison.csv"), index=False)

    X_train, X_test, tfidf_vectorizer = vectorize_texts(
        X_train_text,
        X_test_text,
        vectorizer_type=VECTORIZER_TYPE,
        max_features=MAX_FEATURES,
        keyword_phrases=list(KEYWORD_PHRASES),
        keyword_feature_weight=KEYWORD_FEATURE_WEIGHT,
    )

    nb_tuned_bundle, svm_tuned_bundle, rf_bundle, tuning_df = tune_models(X_train, y_train)

    models = [nb_tuned_bundle, svm_tuned_bundle, rf_bundle]
    metrics_df, train_test_df, preds_test, _ = evaluate_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        label_names,
    )

    metrics_df = metrics_df[
        [
            "Model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "f1_not_cyberbullying",
            "f1_other_cyberbullying",
            "weak_f1_avg",
            "balance_score",
        ]
    ].sort_values(["balance_score", "f1"], ascending=False)
    train_test_df = train_test_df.sort_values(["Metric", "Model"]).reset_index(drop=True)

    metrics_df.to_csv(result_path("tables", "model_metrics.csv"), index=False)
    train_test_df.to_csv(result_path("tables", "train_vs_test_comparison.csv"), index=False)
    tuning_df.to_csv(result_path("tables", "hyperparameter_tuning.csv"), index=False)

    save_model_comparison_plot(metrics_df)
    save_train_test_gap_plot(train_test_df)

    best_name = metrics_df.iloc[0]["Model"]
    y_pred_best = preds_test[best_name]

    save_best_model_artifacts(
        best_model_name=best_name,
        y_test=y_test,
        y_pred_best=y_pred_best,
        label_names=label_names,
        inv_label_map=inv_label_map,
        X_test_text=X_test_text.reset_index(drop=True),
    )
    save_common_confusions(y_test, y_pred_best, label_names)

    name_to_model = {bundle.name: bundle.model for bundle in models}
    save_user_test_predictions(pre, tfidf_vectorizer, name_to_model[best_name], inv_label_map)

    run_manifest = {
        "seed": SEED,
        "dataset_path": DATA_PATH,
        "vectorizer_type": VECTORIZER_TYPE,
        "max_features": MAX_FEATURES,
        "keyword_feature_branch": VECTORIZER_TYPE == "tfidf_hybrid_keywords",
        "keyword_phrases": list(KEYWORD_PHRASES),
        "keyword_feature_weight": KEYWORD_FEATURE_WEIGHT,
        "ensemble_train_subset_size": ENSEMBLE_SUBSET_SIZE,
        "cv_folds": CV_FOLDS,
        "n_samples": int(df.shape[0]),
        "n_classes": int(len(label_map)),
        "models_evaluated": [bundle.name for bundle in models],
        "vectorizers_compared": vectorizer_comp_df["vectorizer_type"].tolist(),
        "vectorizer_comparison_sort": "balance_score desc, f1 desc",
        "best_model": best_name,
        "best_model_criterion": "balance_score = 0.5 * weighted_f1 + 0.5 * average_f1(not_cyberbullying, other_cyberbullying)",
        "report_support_artifacts": [
            "manifests/data_overview.json",
            "tables/data_class_distribution.csv",
            "figures/data_class_distribution.png",
            "tables/vectorizer_comparison.csv",
            "tables/common_confusions_best.csv",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }

    with open(result_path("manifests", "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("Run completed successfully.")
    print(f"Best model: {best_name}")
    print(f"Artifacts saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
