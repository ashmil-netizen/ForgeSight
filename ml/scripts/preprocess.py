"""Clean the AI4I data, create a stratified split, and save preprocessing artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ML_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ML_DIR / "data" / "raw" / "ai4i_2020_raw.csv"
PROCESSED_DIR = ML_DIR / "data" / "processed"
ARTIFACT_DIR = ML_DIR / "artifacts"
TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps to be fitted on training data only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def class_distribution(target: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = target.value_counts().sort_index()
    percentages = target.value_counts(normalize=True).sort_index() * 100
    return {
        str(label): {
            "count": int(count),
            "percentage": round(float(percentages.loc[label]), 4),
        }
        for label, count in counts.items()
    }


def main() -> None:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. Complete Phase 1 first."
        )

    original = pd.read_csv(RAW_DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing_columns = required.difference(original.columns)
    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    source_rows = len(original)
    missing_before = int(original[FEATURES + [TARGET]].isna().sum().sum())
    cleaned = original[FEATURES + [TARGET]].copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = source_rows - len(cleaned)

    X = cleaned[FEATURES]
    y = cleaned[TARGET]
    if y.isna().any():
        raise ValueError(
            "The target contains missing values. Resolve target labels before splitting."
        )
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only binary 0/1 labels.")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two records for stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
    )
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
    )

    train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    train_transformed[TARGET] = y_train.reset_index(drop=True)
    train_transformed.to_csv(PROCESSED_DIR / "train_transformed.csv", index=False)

    test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
    test_transformed[TARGET] = y_test.reset_index(drop=True)
    test_transformed.to_csv(PROCESSED_DIR / "test_transformed.csv", index=False)

    pipeline_path = ARTIFACT_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_path)

    train_counts = y_train.value_counts().sort_index()
    class_weights = {
        str(label): round(len(y_train) / (len(train_counts) * int(count)), 8)
        for label, count in train_counts.items()
    }
    summary = {
        "source": "UCI AI4I 2020 Predictive Maintenance Dataset (ID 601)",
        "source_file": "data/raw/ai4i_2020_raw.csv",
        "original_rows": source_rows,
        "cleaned_rows": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "missing_values_in_selected_features_and_target_before_cleaning": missing_before,
        "missing_value_handling": (
            "Numeric median and categorical most-frequent imputers are fitted "
            "on training data only."
        ),
        "target": TARGET,
        "input_features": FEATURES,
        "excluded_from_model_inputs": [
            column
            for column in original.columns
            if column not in FEATURES + [TARGET]
        ],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_class_distribution": class_distribution(y_train),
            "test_class_distribution": class_distribution(y_test),
        },
        "preprocessing": {
            "numeric": "median imputation then standard scaling",
            "categorical": "most-frequent imputation then one-hot encoding",
            "unknown_categories": "ignored",
            "fit_on": "training features only",
            "output_feature_names": feature_names,
        },
        "imbalance_handling": {
            "method": "balanced class weights calculated from training labels only",
            "class_weights": class_weights,
            "note": (
                "No resampling was applied before splitting. Pass these weights "
                "to the classifier in Phase 4; keep the test set at its natural "
                "class distribution for evaluation."
            ),
        },
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "preprocessor_file": "artifacts/preprocessor.joblib",
    }
    summary_path = ARTIFACT_DIR / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Raw dataset rows: {source_rows}")
    print(f"Cleaned dataset rows: {len(cleaned)}")
    print(f"Exact duplicate rows removed: {duplicates_removed}")
    print(f"Missing values in selected features/target before cleaning: {missing_before}")
    print(f"Input features: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"Train shape: {len(X_train)} rows, {len(feature_names)} transformed features")
    print(f"Test shape: {len(X_test)} rows, {len(feature_names)} transformed features")
    print(f"Training class weights: {class_weights}")
    print(f"Processed datasets saved to: {PROCESSED_DIR}")
    print(f"Fitted preprocessor saved to: {pipeline_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
