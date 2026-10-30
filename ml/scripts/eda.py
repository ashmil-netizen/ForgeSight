"""Explore the original UCI AI4I dataset and save a findings report and charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ML_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ML_DIR / "data" / "raw" / "ai4i_2020_raw.csv"
REPORT_DIR = ML_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

sns.set_theme(style="whitegrid", palette="deep")


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {DATA_PATH}. Complete Phase 1 first."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = {
        "Type",
        TARGET,
        *NUMERIC_FEATURES,
        *FAILURE_MODE_COLUMNS,
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The raw dataset is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    failure_counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    failure_percent = failure_counts / len(data) * 100
    parameter_summary = data.groupby(TARGET)[NUMERIC_FEATURES].agg(
        ["mean", "median", "std"]
    )
    report_parameter_summary = pd.DataFrame(
        [
            {
                "Parameter": feature,
                "No-failure mean": parameter_summary.loc[0, (feature, "mean")],
                "Failure mean": parameter_summary.loc[1, (feature, "mean")],
                "No-failure median": parameter_summary.loc[0, (feature, "median")],
                "Failure median": parameter_summary.loc[1, (feature, "median")],
            }
            for feature in NUMERIC_FEATURES
        ]
    )
    correlations = data[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)
    type_failure_rate = data.groupby("Type", observed=False)[TARGET].agg(
        ["sum", "count", "mean"]
    )
    type_failure_rate["mean"] *= 100

    print(f"Dataset shape: {data.shape}")
    print("\nDescriptive statistics:")
    print(data.describe(include="all").to_string())
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print(f"\nDuplicate rows: {data.duplicated().sum()}")
    print("\nMachine failure counts:")
    print(failure_counts.to_string())
    print("\nMachine failure percentages:")
    print(failure_percent.round(2).to_string())
    print("\nOperating parameter summary by machine failure (0 = no, 1 = yes):")
    print(parameter_summary.round(2).to_string())
    print("\nFailure rate by machine Type (descriptive, not causal):")
    print(type_failure_rate.round(2).to_string())
    print("\nNumeric feature correlations with Machine failure:")
    print(correlations[TARGET].drop(TARGET).sort_values(ascending=False).to_string())

    plt.figure(figsize=(7, 5))
    count_plot = sns.barplot(
        x=failure_counts.index.astype(str),
        y=failure_counts.values,
        hue=failure_counts.index.astype(str),
        legend=False,
    )
    count_plot.set(
        title="Machine failure class distribution",
        xlabel="Machine failure (0 = no, 1 = yes)",
        ylabel="Record count",
    )
    save_figure("failure_class_distribution.png")

    grouped = data.melt(
        id_vars=TARGET,
        value_vars=NUMERIC_FEATURES,
        var_name="Operating parameter",
        value_name="Value",
    )
    grid = sns.catplot(
        data=grouped,
        x=TARGET,
        y="Value",
        col="Operating parameter",
        col_wrap=3,
        kind="box",
        sharey=False,
        height=3.4,
        aspect=1.15,
    )
    grid.set_axis_labels("Machine failure (0 = no, 1 = yes)", "Value")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Operating parameters by machine failure label", y=1.03)
    grid.figure.savefig(
        FIGURE_DIR / "parameters_by_failure.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=type_failure_rate.reset_index(),
        x="Type",
        y="mean",
        hue="Type",
        legend=False,
    )
    plt.title("Observed machine failure rate by product Type")
    plt.xlabel("Product Type")
    plt.ylabel("Records labeled as failure (%)")
    save_figure("failure_rate_by_type.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
    )
    plt.title("Correlation of numeric operating parameters and failure label")
    save_figure("numeric_correlation_heatmap.png")

    positives = int(failure_counts.loc[1])
    negatives = int(failure_counts.loc[0])
    positive_rate = float(failure_percent.loc[1])
    missing_total = int(data.isna().sum().sum())
    duplicate_total = int(data.duplicated().sum())
    sorted_correlations = correlations[TARGET].drop(TARGET).sort_values(
        ascending=False
    )
    strongest_feature = sorted_correlations.index[0]
    strongest_correlation = sorted_correlations.iloc[0]
    failure_mode_counts = {
        column: int(data[column].sum()) for column in FAILURE_MODE_COLUMNS
    }

    report = f"""# Phase 2: Exploratory Data Analysis

## Dataset overview

- Source file: `data/raw/ai4i_2020_raw.csv` (original Phase 1 download; not modified).
- Shape: {len(data):,} rows by {data.shape[1]} columns.
- Missing cells: {missing_total}.
- Exact duplicate rows: {duplicate_total}.
- The identifier columns `UDI` and `Product ID` are present in this raw dataset. Their relevance will be addressed during preprocessing, not changed in this EDA phase.

## Target balance

| Machine failure label | Records | Share |
|---|---:|---:|
| 0 (no failure) | {negatives:,} | {failure_percent.loc[0]:.2f}% |
| 1 (failure) | {positives:,} | {positive_rate:.2f}% |

The failure class is the minority class. Accuracy alone would therefore be a poor measure of whether a future classifier detects failures.

![Machine failure class distribution](figures/failure_class_distribution.png)

**What it shows:** the count difference between records labeled as failure and non-failure. It makes class imbalance visible; it is not a model result.

## Operating parameters

The table below compares descriptive statistics by the binary `Machine failure` label. These are associations in this dataset, not evidence that a parameter caused failure.

{report_parameter_summary.round(2).to_markdown(index=False)}

![Operating parameters grouped by machine failure label](figures/parameters_by_failure.png)

**What it shows:** each box plot compares the distribution of one operating parameter for records labeled no-failure and failure. Different vertical scales are used so each parameter's spread can be seen. Overlap between boxes indicates that a single parameter may not cleanly distinguish the classes.

## Product Type

{type_failure_rate.round(2).to_markdown()}

![Observed failure rate by product Type](figures/failure_rate_by_type.png)

**What it shows:** the share of records labeled as failure within each product Type. The rates are descriptive and may be affected by sample counts; they do not establish that Type causes failure.

## Numeric correlations

{sorted_correlations.to_frame("Correlation with Machine failure").round(4).to_markdown()}

![Numeric correlation heatmap](figures/numeric_correlation_heatmap.png)

**What it shows:** pairwise linear correlations among numeric operating parameters and the binary failure label. Correlation is not causation, and a weak linear correlation does not rule out nonlinear or interacting predictive patterns.

## Failure mode labels

The raw dataset includes the following failure mode label counts: `{failure_mode_counts}`. These labels describe failure types; they must not be used as input features when predicting `Machine failure`, because that would leak outcome information.

## Findings

1. The dataset contains {len(data):,} records and {data.shape[1]} columns; this analysis left the source CSV unchanged.
2. There are {missing_total} missing cells and {duplicate_total} exact duplicate rows in the loaded data.
3. {positive_rate:.2f}% of records are labeled as machine failures ({positives:,} of {len(data):,}), confirming substantial class imbalance.
4. The largest positive linear correlation among the measured numeric parameters is for `{strongest_feature}` ({strongest_correlation:.3f}). This is a descriptive statistic, not a feature-selection decision or causal finding.
5. In later modeling, use `Machine failure` as the target and exclude identifiers and failure-mode labels from the predictor inputs. Assess failure-class recall and precision-recall performance alongside other metrics.

## Generated charts

- `figures/failure_class_distribution.png`
- `figures/parameters_by_failure.png`
- `figures/failure_rate_by_type.png`
- `figures/numeric_correlation_heatmap.png`
"""
    report_path = REPORT_DIR / "phase2_eda.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nEDA report saved to: {report_path}")
    print(f"Charts saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
