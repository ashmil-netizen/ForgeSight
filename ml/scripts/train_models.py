"""Train, compare, and evaluate class-weighted failure classifiers."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from preprocess import FEATURES, TARGET, build_preprocessor


ML_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ML_DIR / "data" / "processed"
ARTIFACT_DIR = ML_DIR / "artifacts"
REPORT_DIR = ML_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def positive_class_probability(model: Pipeline, X: pd.DataFrame) -> list[float]:
    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    return model.predict_proba(X)[:, positive_index].tolist()


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | int], list[float], list[int]]:
    probabilities = positive_class_probability(model, X_test)
    predictions = [int(score >= DECISION_THRESHOLD) for score in probabilities]
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
    }
    return metrics, probabilities, predictions


def save_confusion_matrices(
    y_test: pd.Series,
    predictions: dict[str, list[int]],
) -> dict[str, list[list[int]]]:
    names = list(predictions)
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(11, 4.5),
        constrained_layout=True,
    )
    matrices: dict[str, list[list[int]]] = {}
    for axis, name in zip(axes, names):
        matrix = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        matrices[name] = matrix.tolist()
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="black",
                )
        axis.set(
            title=name,
            xlabel="Predicted label",
            ylabel="Actual label",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No failure", "Failure"],
            yticklabels=["No failure", "Failure"],
        )
    figure.colorbar(image, ax=axes, fraction=0.035, pad=0.04, label="Records")
    figure.suptitle("Held-out test confusion matrices (threshold = 0.50)")
    figure.savefig(FIGURE_DIR / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return matrices


def save_curves(
    y_test: pd.Series,
    probabilities: dict[str, list[float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, scores in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        axes[0].plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc_score(y_test, scores):.3f})",
        )
        axes[1].plot(
            recall,
            precision,
            label=f"{name} (AP = {average_precision_score(y_test, scores):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    axes[0].set(
        title="ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
    )
    axes[1].axhline(
        y=float(y_test.mean()),
        linestyle="--",
        color="grey",
        label=f"Failure prevalence = {y_test.mean():.3f}",
    )
    axes[1].set(
        title="Precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis in axes:
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "roc_and_precision_recall.png", dpi=160)
    plt.close(figure)


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    summary_path = ARTIFACT_DIR / "preprocessing_summary.json"
    for path in (train_path, test_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Phase 3 artifact not found: {path}. "
                "Run scripts/preprocess.py first."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    preprocessing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        preprocessing_summary.get("target") != TARGET
        or preprocessing_summary.get("input_features") != FEATURES
    ):
        raise ValueError("Phase 3 preprocessing summary does not match model inputs.")
    expected_columns = set(FEATURES + [TARGET])
    if set(train.columns) != expected_columns or set(test.columns) != expected_columns:
        raise ValueError(
            "Train/test columns do not match the Phase 3 feature and target schema."
        )

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both train and test splits must contain both target classes.")

    models = build_models()
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores: dict[str, list[float]] = {}
    for name, classifier in models.items():
        pipeline = build_pipeline(classifier)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="average_precision",
            cv=cross_validation,
            n_jobs=1,
            error_score="raise",
        )
        cv_scores[name] = [float(score) for score in scores]

    # Select by training-only cross-validation; the held-out test scores do not
    # influence the choice of model.
    selected_name = max(
        cv_scores,
        key=lambda name: sum(cv_scores[name]) / len(cv_scores[name]),
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fitted_models: dict[str, Pipeline] = {}
    metrics_by_model: dict[str, dict[str, float | int]] = {}
    probabilities_by_model: dict[str, list[float]] = {}
    predictions_by_model: dict[str, list[int]] = {}
    for name, classifier in models.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        metrics, probabilities, predictions = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )
        fitted_models[name] = pipeline
        metrics_by_model[name] = metrics
        probabilities_by_model[name] = probabilities
        predictions_by_model[name] = predictions

    confusion_matrices = save_confusion_matrices(y_test, predictions_by_model)
    save_curves(y_test, probabilities_by_model)
    joblib.dump(fitted_models[selected_name], ARTIFACT_DIR / "best_model.joblib")

    rows = []
    for name in models:
        row: dict[str, float | str] = {
            "model": name,
            "cv_average_precision_mean": sum(cv_scores[name]) / CV_FOLDS,
            "cv_average_precision_std": pd.Series(cv_scores[name]).std(ddof=0),
        }
        row.update(metrics_by_model[name])
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        "cv_average_precision_mean",
        ascending=False,
    )
    comparison.to_csv(REPORT_DIR / "phase4_model_comparison.csv", index=False)

    matrix_details = {
        name: {
            "true_negatives": matrix[0][0],
            "false_positives": matrix[0][1],
            "false_negatives": matrix[1][0],
            "true_positives": matrix[1][1],
        }
        for name, matrix in confusion_matrices.items()
    }
    selected_cv_mean = sum(cv_scores[selected_name]) / CV_FOLDS
    report_lines = [
        "# Phase 4: Model Training and Evaluation",
        "",
        "## Method",
        "",
        "- Models: class-weighted Logistic Regression baseline and Random Forest.",
        f"- Selection criterion: mean {CV_FOLDS}-fold stratified cross-validation average precision on the training split only.",
        f"- Selected model: **{selected_name}** (training CV average precision {selected_cv_mean:.4f}).",
        "- Both pipelines fit imputers, scaling, and one-hot encoding inside each CV fold, then refit on the complete training split.",
        "- The held-out test split was evaluated once after model selection. No hyperparameter search or test-based threshold tuning was performed.",
        f"- Test decision threshold: {DECISION_THRESHOLD:.2f}. Class weights are used to address imbalance; their probability outputs should not be assumed calibrated.",
        "",
        "## Model comparison",
        "",
        comparison.round(4).to_markdown(index=False),
        "",
        "CV values are training-only. Test metrics use the untouched held-out split; positive-class recall and precision-recall average precision are important because failures are rare.",
        "",
        "## Confusion matrices and error interpretation",
        "",
        f"![Held-out confusion matrices](figures/confusion_matrices.png)",
        "",
    ]
    for name, counts in matrix_details.items():
        report_lines.extend(
            [
                f"### {name}",
                "",
                f"- True negatives: {counts['true_negatives']}",
                f"- False positives: {counts['false_positives']} — healthy records flagged for follow-up, which can create unnecessary inspections.",
                f"- False negatives: {counts['false_negatives']} — failures missed by the model, which may be more consequential in a maintenance setting.",
                f"- True positives: {counts['true_positives']}",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Ranking curves",
            "",
            "![Held-out ROC and precision-recall curves](figures/roc_and_precision_recall.png)",
            "",
            "ROC-AUC measures ranking across thresholds; average precision summarizes the precision-recall curve and is more informative than accuracy for a rare positive class. The chosen default threshold is not a deployment recommendation.",
            "",
            "## Artifacts",
            "",
            "- `artifacts/best_model.joblib`: selected fitted preprocessing-plus-classifier pipeline.",
            "- `artifacts/training_summary.json`: CV scores, held-out metrics, confusion counts, selection criterion, and versions.",
            "- `reports/phase4_model_comparison.csv`: CV and held-out metric comparison.",
            "",
            "This is an educational prototype on a public dataset, not a validated industrial safety system.",
        ]
    )
    report_path = REPORT_DIR / "phase4_model_evaluation.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    training_summary = {
        "source": "UCI AI4I 2020 Predictive Maintenance Dataset (ID 601)",
        "selection": {
            "criterion": f"highest mean {CV_FOLDS}-fold stratified training average precision",
            "selected_model": selected_name,
            "selected_model_cv_average_precision_mean": selected_cv_mean,
            "test_set_used_for_selection": False,
            "threshold_tuned_on_test": False,
            "decision_threshold": DECISION_THRESHOLD,
        },
        "cross_validation_average_precision": {
            name: {
                "fold_scores": scores,
                "mean": sum(scores) / len(scores),
                "std": float(pd.Series(scores).std(ddof=0)),
            }
            for name, scores in cv_scores.items()
        },
        "held_out_test_metrics": metrics_by_model,
        "held_out_confusion_matrices": confusion_matrices,
        "error_counts": matrix_details,
        "imbalance_handling": {
            "logistic_regression": "class_weight='balanced', calculated by the estimator on each fit",
            "random_forest": "class_weight='balanced_subsample', calculated per bootstrap sample",
        },
        "test_rows": len(y_test),
        "test_positive_count": int(y_test.sum()),
        "versions": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "model_file": "artifacts/best_model.joblib",
        "note": (
            "Class-weighted model scores are not guaranteed to be calibrated "
            "probabilities. The dataset is an educational benchmark and does "
            "not establish real-world industrial performance."
        ),
    }
    (ARTIFACT_DIR / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )

    print(f"Training-only CV selection: {selected_name}")
    print(f"CV average precision: {selected_cv_mean:.4f}")
    print(f"Decision threshold used for test comparison: {DECISION_THRESHOLD:.2f}")
    print("\nHeld-out test comparison:")
    print(comparison.round(4).to_string(index=False))
    print(f"\nBest model saved to: {ARTIFACT_DIR / 'best_model.joblib'}")
    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
