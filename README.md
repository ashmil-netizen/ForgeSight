# ForgeSight — Predictive Maintenance Demo

ForgeSight is an educational predictive-maintenance application built around
the public [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
It combines a trained machine-learning model, a local REST API, and a web
dashboard.

> **Educational use only.** ForgeSight is not connected to industrial
> equipment or live sensors. Its scores, recommendations, and example data
> are not a diagnosis and must not be used to make operational or safety
> decisions.

## What it does

- Presents dataset facts, evaluation results, and benchmark distributions.
- Lets you enter machine conditions or fill the form with one of three
  real-record presets.
- Sends the inputs to a local FastAPI service, which returns a Random Forest
  prediction, model score, risk band, and separately identified
  rule-based guidance.
- Stores predictions in a local SQLite database and displays prediction
  history.
- Includes a Logistic Regression baseline for model comparison.

The presets only fill the form; they do not submit a prediction. A preset's
failure label describes its original dataset record and is not a promise about
what the trained model will predict.

## Project structure

```text
ml/
  api/                 FastAPI endpoints and local SQLite prediction history
  artifacts/           Trained Random Forest and training summaries
  data/raw/            UCI AI4I source CSVs
  data/processed/      Cleaned, split, and transformed datasets
  reports/             EDA, model evaluation, and generated charts
  scripts/             Dataset inspection, EDA, preprocessing, and training
  tests/               Backend prediction and API tests
  prediction.py        Input validation, model inference, and guidance rules
frontend/
  src/                 React dashboard, API client, and styles
  package.json         Frontend scripts and dependencies
```

## Run it locally

### Prerequisites

- Git
- Python 3.10 or later (Python 3.14 was used to generate the included model)
- Node.js 18 or later and npm

The repository includes the trained model and AI4I CSV files, so you **do not**
need to download the dataset or train a model just to start the application.
The first install does need internet access to install Python and npm packages.

### 1. Clone the repository

Open a terminal and run:

```shell
git clone https://github.com/ashmil-netizen/ForgeSight.git
cd ForgeSight
```

### 2. Install the backend dependencies

**Windows PowerShell:**

```powershell
cd ml
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS or Linux:**

```shell
cd ml
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

If `py` is not available on Windows, use `python` to create the virtual
environment. Python package versions can affect loading serialized
scikit-learn models; the included training summary records the versions used.
If you see a scikit-learn model-version warning, recreate the model using the
optional training steps below in your environment.

### 3. Start the backend

Keep this first terminal open while you use the application.

**Windows PowerShell**, from the `ml` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --app-dir .
```

**macOS or Linux**, from the `ml` directory:

```shell
./.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --app-dir .
```

When the backend is ready, its health endpoint is
<http://127.0.0.1:8000/api/health>. It should return a status of `ok` and
`model` of `loaded`.

### 4. Install and start the frontend

Open a **second terminal** at the repository root, then run:

```shell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173/> in your browser. Go to **Machine analysis** to
try an example or enter your own benchmark-style parameters. Keep both
terminals running; stop a server with `Ctrl+C`.

## API endpoints

The local API is served at `http://127.0.0.1:8000`:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Check API and model status |
| `GET` | `/api/model-info` | Read model details and evaluation information |
| `POST` | `/api/predict` | Validate inputs, predict, and save the result |
| `GET` | `/api/predictions` | Read saved prediction history |

Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

The frontend uses `http://127.0.0.1:8000/api` by default. To use another API
base URL, set `VITE_API_BASE_URL` in the frontend environment before starting
Vite.

Prediction history is stored locally in `ml/data/predictions.sqlite3`. Set
`PREDICTIONS_DB` to choose another database path. The local database is
excluded from Git.

## Machine-learning workflow

The target is `Machine failure`. Identifier columns (`UDI`, `Product ID`) and
failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from
model inputs to avoid target leakage. The dataset contains 10,000 rows and
339 machine-failure labels (3.39%).

Model selection compares class-weighted Logistic Regression and Random Forest
using five-fold stratified cross-validation average precision on the training
data only. Random Forest was selected with mean cross-validation average
precision **0.7495**. On the held-out test split at a 0.50 decision threshold,
it achieved **0.441 recall**, **0.938 precision**, **0.600 F1**, **0.966
ROC-AUC**, and **0.780 average precision**. It missed 38 of the 68 failure
records in that test split. The Logistic Regression baseline achieved 0.824
recall but 0.142 precision.

These metrics describe one benchmark split, not expected real-world
performance. The class-weighted model score is not guaranteed to be a
calibrated probability. Risk bands and maintenance suggestions are educational
heuristics, not validated equipment limits.

### Optional: reproduce the analysis and train a model

The dataset and trained model are already included; these steps are optional.
Run commands from the `ml` directory after installing dependencies:

```shell
python scripts/eda.py
python scripts/preprocess.py
python scripts/train_models.py
```

To fetch the source data again from UCI, run
`python scripts/inspect_dataset.py` first. That step requires internet access.
The scripts write processed data, charts, reports, and model artifacts under
`ml/`.

## Run tests and build

**Backend tests**, from the `ml` directory:

```shell
python -m unittest discover -s tests -v
```

**Frontend production build**, from the `frontend` directory:

```shell
npm ci
npm run build
```

On Windows, if the virtual environment is not activated, use
`.\.venv\Scripts\python.exe` instead of `python` for the backend commands.
On macOS or Linux, use `./.venv/bin/python` if needed.

## Troubleshooting

- **“Could not reach the API”** — make sure the backend terminal is still
  running and `http://127.0.0.1:8000/api/health` reports `model: loaded`.
- **Port already in use** — stop the existing process using port 8000 or 5173,
  or configure a different port and update the frontend API base URL if the
  backend port changes.
- **Model file missing** — confirm `ml/artifacts/best_model.joblib` exists. If
  it does not, run the optional preprocessing and training steps.
- **Python command not found** — install Python and ensure its launcher is
  available in your terminal; on Windows, `py` is commonly available.
- **Frontend dependencies are missing** — run `npm ci` from `frontend`.

## Data source and limitations

This project uses the public
[UCI AI4I 2020 Predictive Maintenance Dataset (ID 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
The benchmark CSVs and held-out evaluation do not represent live telemetry.
Model performance, thresholds, and recommendations have not been validated
for an industrial setting. Do not rely on this prototype for maintenance,
production, or safety decisions.
