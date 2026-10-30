"""REST API for the educational AI4I predictive maintenance model."""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from prediction import (
    INPUT_RANGES,
    INPUT_FEATURES,
    MAINTENANCE_THRESHOLDS,
    RISK_THRESHOLDS,
    CLASSIFICATION_THRESHOLD,
    PredictionService,
)


ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ML_DIR / "data" / "predictions.sqlite3"
DB_PATH = Path(os.environ.get("PREDICTIONS_DB", str(DEFAULT_DB_PATH)))
RAW_DATA_PATH = ML_DIR / "data" / "raw" / "ai4i_2020_raw.csv"
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_type: Literal["L", "M", "H"]
    air_temperature: float = Field(ge=295.3, le=304.5, allow_inf_nan=False)
    process_temperature: float = Field(ge=305.7, le=313.8, allow_inf_nan=False)
    rotational_speed: int = Field(ge=1168, le=2886)
    torque: float = Field(ge=3.8, le=76.6, allow_inf_nan=False)
    tool_wear: int = Field(ge=0, le=253)

    def as_model_input(self) -> dict[str, str | float]:
        return {
            "Type": self.machine_type,
            "Air temperature": self.air_temperature,
            "Process temperature": self.process_temperature,
            "Rotational speed": self.rotational_speed,
            "Torque": self.torque,
            "Tool wear": self.tool_wear,
        }


class PredictionResult(BaseModel):
    id: int
    created_at: str
    predicted_failure: int
    failure_probability: float
    risk_category: str
    decision_threshold: float
    recommendations: list[str]
    parameters: dict[str, str | float]
    model_name: str
    probability_note: str
    risk_thresholds: dict[str, str]


def initialize_history_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_prediction(result: dict) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(DB_PATH)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (created_at, parameters_json, result_json)
            VALUES (?, ?, ?)
            """,
            (
                created_at,
                json.dumps(result["parameters"]),
                json.dumps(result, allow_nan=False),
            ),
        )
        prediction_id = int(cursor.lastrowid)
        connection.commit()
    return {"id": prediction_id, "created_at": created_at, **result}


def get_predictions(limit: int) -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, result_json
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {"id": row[0], "created_at": row[1], **json.loads(row[2])}
        for row in rows
    ]


def get_prediction_count() -> int:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        row = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()
    if row is None:
        raise sqlite3.DatabaseError("Could not count prediction history.")
    return int(row[0])


def load_benchmark_profile() -> dict:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"UCI raw dataset is missing at {RAW_DATA_PATH}. Complete Phase 1 first."
        )
    data = pd.read_csv(RAW_DATA_PATH)
    required = set(NUMERIC_FEATURES + ["Machine failure", "Type"])
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The raw benchmark is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    distributions = {}
    for feature in NUMERIC_FEATURES:
        bins = pd.cut(data[feature], bins=8, include_lowest=True).value_counts().sort_index()
        observed_minimum = float(data[feature].min())
        observed_maximum = float(data[feature].max())
        distributions[feature] = [
            {
                "range": (
                    f"{max(interval.left, observed_minimum):g}–"
                    f"{min(interval.right, observed_maximum):g}"
                ),
                "count": int(count),
            }
            for interval, count in bins.items()
        ]
    return {
        "dataset_rows": len(data),
        "dataset_target_counts": {
            str(label): int(count)
            for label, count in data["Machine failure"].value_counts().items()
        },
        "dataset_machine_type_counts": {
            str(label): int(count)
            for label, count in data["Type"].value_counts().items()
        },
        "parameter_distributions": distributions,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService()
    app.state.benchmark_profile = load_benchmark_profile()
    initialize_history_db()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Educational prediction API using the public UCI AI4I 2020 dataset. "
        "It is not connected to industrial machines or real-time sensors."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "prediction_service"):
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")
    return {"status": "ok", "model": "loaded"}


@app.get("/api/model-info")
def model_info(request: Request) -> dict:
    service: PredictionService = request.app.state.prediction_service
    training = service.summary
    selected = training["selection"]["selected_model"]
    profile = request.app.state.benchmark_profile
    return {
        "project": "Educational AI4I predictive maintenance prototype",
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset (ID 601)",
        **profile,
        "prediction_history_count": get_prediction_count(),
        "model": selected,
        "input_features": INPUT_FEATURES,
        "supported_input_ranges": {
            name: {"minimum": bounds[0], "maximum": bounds[1]}
            for name, bounds in INPUT_RANGES.items()
        },
        "supported_machine_types": ["L", "M", "H"],
        "decision_threshold": CLASSIFICATION_THRESHOLD,
        "risk_thresholds": {
            "low_max_exclusive": RISK_THRESHOLDS["low_max_exclusive"],
            "medium_max_exclusive": RISK_THRESHOLDS["medium_max_exclusive"],
            "high_minimum_inclusive": RISK_THRESHOLDS["medium_max_exclusive"],
        },
        "rule_based_maintenance_thresholds": MAINTENANCE_THRESHOLDS,
        "test_metrics": training["held_out_test_metrics"][selected],
        "limitations": [
            "Not connected to industrial machines or real-time sensors.",
            "Class-weighted model scores are not guaranteed to be calibrated probabilities.",
            "Benchmark performance is not evidence of real-world operational or safety performance.",
        ],
    }


@app.post("/api/predict", response_model=PredictionResult)
def predict(payload: PredictionInput, request: Request) -> dict:
    service: PredictionService = request.app.state.prediction_service
    try:
        result = service.predict(payload.as_model_input())
        return save_prediction(result)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction succeeded but could not be saved to history.",
        ) from error


@app.get("/api/predictions", response_model=list[PredictionResult])
def prediction_history(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    if not hasattr(request.app.state, "prediction_service"):
        raise HTTPException(status_code=503, detail="Prediction service is not ready.")
    try:
        return get_predictions(limit)
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Could not read prediction history.",
        ) from error
