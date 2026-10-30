"""Inference and educational maintenance heuristics for the trained AI4I model."""

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "artifacts" / "best_model.joblib"
SUMMARY_PATH = ML_DIR / "artifacts" / "training_summary.json"
INPUT_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

# These are the observed min/max values in the public AI4I benchmark. They keep
# this educational demo from implying validated extrapolation beyond its data.
INPUT_RANGES = {
    "Air temperature": (295.3, 304.5),
    "Process temperature": (305.7, 313.8),
    "Rotational speed": (1168, 2886),
    "Torque": (3.8, 76.6),
    "Tool wear": (0, 253),
}
ALLOWED_TYPES = {"L", "M", "H"}
INTEGER_FEATURES = {"Rotational speed", "Tool wear"}
CLASSIFICATION_THRESHOLD = 0.5
RISK_THRESHOLDS = {"low_max_exclusive": 0.10, "medium_max_exclusive": 0.30}
MAINTENANCE_THRESHOLDS = {
    "Tool wear": {"minimum": 200, "unit": "min"},
    "Torque": {"minimum": 60, "unit": "Nm"},
    "Air temperature": {"minimum": 303, "unit": "K"},
    "Process temperature": {"minimum": 312, "unit": "K"},
}


class PredictionService:
    """Load the selected model and produce predictions plus rule-based guidance."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        summary_path: Path = SUMMARY_PATH,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run scripts/train_models.py."
            )
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Training summary not found at {summary_path}. "
                "Run scripts/train_models.py."
            )
        self.model = joblib.load(model_path)
        self.summary: dict[str, Any] = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

    @staticmethod
    def validate_input(parameters: dict[str, Any]) -> dict[str, Any]:
        missing = set(INPUT_FEATURES).difference(parameters)
        extra = set(parameters).difference(INPUT_FEATURES)
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"Unexpected inputs: {', '.join(sorted(extra))}")
        if parameters["Type"] not in ALLOWED_TYPES:
            raise ValueError("Type must be one of L, M, or H.")

        normalized = dict(parameters)
        for field, (minimum, maximum) in INPUT_RANGES.items():
            try:
                value = float(parameters[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be a number.") from error
            if not math.isfinite(value):
                raise ValueError(f"{field} must be a finite number.")
            if field in INTEGER_FEATURES and not value.is_integer():
                raise ValueError(f"{field} must be an integer.")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{field} must be within the AI4I dataset range "
                    f"{minimum} to {maximum}."
                )
            normalized[field] = value
        return normalized

    @staticmethod
    def risk_category(probability: float) -> str:
        if probability < RISK_THRESHOLDS["low_max_exclusive"]:
            return "Low"
        if probability < RISK_THRESHOLDS["medium_max_exclusive"]:
            return "Medium"
        return "High"

    @staticmethod
    def recommendations(
        parameters: dict[str, Any],
        probability: float,
    ) -> list[str]:
        recommendations: list[str] = []
        if parameters["Tool wear"] >= MAINTENANCE_THRESHOLDS["Tool wear"]["minimum"]:
            recommendations.append(
                "Tool wear is at least 200 minutes: inspect the cutting tool "
                "and consider replacement according to your maintenance procedure."
            )
        if parameters["Torque"] >= MAINTENANCE_THRESHOLDS["Torque"]["minimum"]:
            recommendations.append(
                "Torque is at least 60 Nm: review mechanical loading and inspect "
                "the drive train."
            )
        if (
            parameters["Air temperature"]
            >= MAINTENANCE_THRESHOLDS["Air temperature"]["minimum"]
        ):
            recommendations.append(
                "Air temperature is at least 303 K: review ambient conditions "
                "and inspect cooling or ventilation."
            )
        if (
            parameters["Process temperature"]
            >= MAINTENANCE_THRESHOLDS["Process temperature"]["minimum"]
        ):
            recommendations.append(
                "Process temperature is at least 312 K: review process "
                "temperature control and cooling."
            )
        if probability >= RISK_THRESHOLDS["medium_max_exclusive"]:
            recommendations.append(
                "The model score is in the High risk band: arrange a qualified "
                "inspection; do not treat this educational prediction as a diagnosis."
            )
        elif probability >= RISK_THRESHOLDS["low_max_exclusive"]:
            recommendations.append(
                "The model score is in the Medium risk band: review the operating "
                "record and consider a routine inspection."
            )
        if not recommendations:
            recommendations.append(
                "No configured rule-based alerts were triggered. Continue "
                "maintenance according to the equipment manufacturer's schedule."
            )
        return recommendations

    def predict(self, parameters: dict[str, Any]) -> dict[str, Any]:
        normalized = self.validate_input(parameters)
        frame = pd.DataFrame([normalized], columns=INPUT_FEATURES)
        classifier = self.model.named_steps["classifier"]
        classes = list(classifier.classes_)
        if 1 not in classes:
            raise RuntimeError("The loaded model has no positive failure class.")
        positive_index = classes.index(1)
        probability = float(self.model.predict_proba(frame)[0, positive_index])
        predicted_failure = int(self.model.predict(frame)[0])
        risk = self.risk_category(probability)
        return {
            "predicted_failure": predicted_failure,
            "failure_probability": probability,
            "risk_category": risk,
            "decision_threshold": CLASSIFICATION_THRESHOLD,
            "recommendations": self.recommendations(normalized, probability),
            "parameters": normalized,
            "model_name": self.summary["selection"]["selected_model"],
            "probability_note": (
                "This is a class-weighted model score and is not guaranteed to be "
                "a calibrated probability. It is an educational prediction on "
                "the UCI AI4I benchmark, not a real-time industrial diagnosis."
            ),
            "risk_thresholds": {
                "low": "score < 0.10",
                "medium": "0.10 <= score < 0.30",
                "high": "score >= 0.30",
            },
        }
