# Red Wine Quality: Simple ML Pipeline

A small end-to-end machine learning pipeline that predicts whether a red wine is rated **good** (quality >= 6) from its physicochemical measurements.

## Result

| Metric | Value |
|---|---|
| **Test accuracy** | **73.5%** (200 of 272 test wines correct) |
| Baseline (always predict "good") | 52.9% |

The model beats the majority-class baseline by about 20 percentage points.

Confusion matrix on the test set (rows = actual, columns = predicted):

|  | Predicted not good | Predicted good |
|---|---|---|
| **Actual not good** | 95 | 33 |
| **Actual good** | 39 | 105 |

Precision and recall are 0.71 to 0.76 for both classes, so the model does not favour one class.

> These numbers come from a single 80/20 split with a fixed seed (`random_state=42`). With only 272 test wines, a different split could move accuracy by a few points.

## Data

[Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (Kaggle, `uciml/red-wine-quality-cortez-et-al-2009`), downloaded with `kagglehub`.

- 1,599 red wines, 11 numeric features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
- Original label: `quality`, an integer score from 3 to 8.

## Pipeline

All steps are in [wine_pipeline.ipynb](wine_pipeline.ipynb), which is saved with its outputs.

1. **Setup:** imports and config (dataset version, random seed, feature list) in one place at the top.
2. **Ingestion and schema check:** download the dataset with `kagglehub`, load the CSV, and assert that the expected columns exist and are numeric.
3. **Cleaning:** no missing values; 240 duplicate rows are dropped (1,599 to 1,359 rows) so the same wine cannot appear in both train and test.
4. **Target:** `good = quality >= 6`, giving 719 good and 640 not-good wines (52.9% / 47.1%).
5. **Quick EDA:** summary statistics and correlations with quality. Alcohol has the strongest positive correlation (0.48) and volatile acidity the strongest negative (-0.40).
6. **Split:** stratified 80/20 train/test (1,087 / 272 rows). `quality` is excluded from the features because the target is derived from it.
7. **Model:** scikit-learn `Pipeline` of `StandardScaler` followed by `LogisticRegression`. The scaler sits inside the pipeline, so it is fitted on training data only.
8. **Evaluation:** accuracy, confusion matrix and classification report on the held-out test set.

## Requirements

- **Python 3.11 or newer** (declared as `requires-python = ">=3.11"` in `pyproject.toml`). Developed and tested on Python 3.14; 3.11 is the declared minimum and has not been run.
- Internet access on the first run, to download the dataset from Kaggle (no Kaggle account or API key needed).

## Run it

Run all cells from the top once the notebook is open. You can also open it in VS Code and pick the project's `.venv` kernel.

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the right Python version and the locked dependencies for you.

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is already installed (or: brew install uv)
uv sync
uv run jupyter lab wine_pipeline.ipynb
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # skip if uv is already installed
uv sync
uv run jupyter lab wine_pipeline.ipynb
```

### Option B: plain pip

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt jupyterlab
.venv/bin/jupyter lab wine_pipeline.ipynb
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt jupyterlab
.venv\Scripts\jupyter lab wine_pipeline.ipynb
```

> Tested on macOS: installing from `uv.lock` and `requirements.txt` and executing the notebook end to end. Not run: the `jupyter lab` launch and the uv installer one-liners. The Windows commands use the standard equivalents (`.venv\Scripts` instead of `.venv/bin`, `py -3` instead of `python3`) but have not been run on a Windows machine.

Dependencies are pinned in `pyproject.toml` (locked in `uv.lock`) and `requirements.txt`, and the dataset version is pinned in the notebook's config cell, so a rerun reproduces the numbers above.

If you change a dependency, update `pyproject.toml`, run `uv lock`, and keep `requirements.txt` in sync.

## Limitations

- Binarizing quality throws away detail; predicting the exact 3 to 8 score would be a harder problem.
- One model, one split, no hyperparameter tuning or cross-validation.
