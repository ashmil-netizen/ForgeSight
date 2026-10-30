"""Download and inspect the UCI AI4I 2020 Predictive Maintenance Dataset."""

from pathlib import Path
from pprint import pprint

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 601
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets
    original = dataset.data.original

    if features is None or targets is None or original is None:
        raise RuntimeError(
            "UCI dataset did not provide the original data, features, and targets."
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset: UCI AI4I 2020 Predictive Maintenance Dataset (ID 601)")
    print(f"Original dataset shape: {original.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print("\nFirst five feature records:")
    print(features.head().to_string(index=False))
    print("\nFirst five target records:")
    print(targets.head().to_string(index=False))
    print("\nFeature columns and data types:")
    print(features.dtypes.to_string())
    print("\nTarget columns and data types:")
    print(targets.dtypes.to_string())
    print("\nDataset metadata:")
    pprint(dataset.metadata)
    print("\nVariable information and descriptions:")
    print(dataset.variables.to_string(index=False))

    features.to_csv(RAW_DATA_DIR / "ai4i_2020_features.csv", index=False)
    targets.to_csv(RAW_DATA_DIR / "ai4i_2020_targets.csv", index=False)
    original.to_csv(RAW_DATA_DIR / "ai4i_2020_raw.csv", index=False)
    print(f"\nUnmodified CSV files saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
