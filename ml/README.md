# Predictive Maintenance ML Component

Phase 1 retrieves and inspects the UCI AI4I 2020 Predictive Maintenance
Dataset (dataset ID 601). The downloaded data is saved without preprocessing
under `data/raw/`. Phase 2 analyses that original CSV and writes a findings
report and charts under `reports/`.

## Run

From the `ml` directory, activate the virtual environment and run:

```powershell
python scripts/inspect_dataset.py
```

The script prints the dataset metadata, variable descriptions, shapes, sample
records, and feature data types. It saves the complete dataset and separate
feature and target CSV files in `data/raw/`.

## Phase 2: exploratory data analysis

Install the EDA plotting dependencies if they are not installed yet, then run
from the `ml` directory:

```powershell
python -m pip install -r requirements.txt
python scripts/eda.py
```

The script reads `data/raw/ai4i_2020_raw.csv` without modifying it. It reports
summary statistics, missing values, duplicates, failure balance, parameter
distributions, and numeric correlations. It writes the findings to
`reports/phase2_eda.md` and PNG charts to `reports/figures/`.

## Phase 3: cleaning and preprocessing

Install the machine-learning dependencies if they are not installed yet, then
run from the `ml` directory:

```powershell
python -m pip install -r requirements.txt
python scripts/preprocess.py
```

The script removes identifiers and failure-mode labels from model inputs,
removes exact duplicate records if present, creates a stratified train/test
split, and fits the reusable preprocessing pipeline on the training data only.
It writes cleaned and split CSVs to `data/processed/`, the fitted pipeline to
`artifacts/preprocessor.joblib`, and a summary (including training-only class
weights) to `artifacts/preprocessing_summary.json`. No oversampling is done
before the split; later model training can use the recorded training-only
weights to address the minority failure class without contaminating evaluation.

## Phase 4: model training and evaluation

Run from the `ml` directory after Phase 3:

```powershell
python scripts/train_models.py
```

The script compares class-weighted Logistic Regression and Random Forest using
stratified cross-validation average precision on the training data only. It
selects the model by that cross-validation score, then evaluates both models
once on the held-out test data. It writes the selected fitted pipeline to
`artifacts/best_model.joblib`, metrics to `artifacts/training_summary.json`,
the comparison table to `reports/phase4_model_comparison.csv`, and evaluation
charts and explanations to `reports/`.

## Phase 5: prediction module

The reusable inference service is in `prediction.py`. It loads the selected
model and training summary, validates parameters against the observed AI4I data
ranges, returns the model prediction and score, and generates separately
identified rule-based maintenance guidance. Risk bands are documented
heuristics: Low `< 0.10`, Medium `0.10–<0.30`, and High `>= 0.30`. The
classifier's failure/no-failure decision still uses its `0.50` threshold.
The educational recommendation rules trigger at tool wear `>= 200 min`, torque
`>= 60 Nm`, air temperature `>= 303 K`, and process temperature `>= 312 K`.
These thresholds are illustrative flags, not validated equipment limits.

## Phase 6: FastAPI backend

From the `ml` directory, start the API with:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive endpoint
documentation is at `http://127.0.0.1:8000/docs`. Endpoints are `GET
/api/health`, `GET /api/model-info`, `POST /api/predict`, and `GET
/api/predictions`. Prediction history is stored in local SQLite at
`data/predictions.sqlite3`; set `PREDICTIONS_DB` to override that location for
testing. This prototype is not a real-time sensor integration.

## Frontend

The React/Vite dashboard is in `../frontend`. Install frontend packages, start
the API as described above, then run:

```powershell
cd ..\frontend
npm install
npm run dev
```

The UI expects the API at `http://127.0.0.1:8000/api` by default. Override it
with a `VITE_API_BASE_URL` environment variable when needed.
