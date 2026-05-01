# Credit card fraud — experiment pipeline

Python pipeline for comparing classical classifiers (logistic regression, random forest, XGBoost) with spiking neural network baselines on a subsample of the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

## Contents

| File | Purpose |
|------|---------|
| `experiment_ccfraud.py` | Loads data, runs feature selection, trains/evaluates models, writes `results/ccfraud_results.json` |
| `viz_ccfraud.py` | Builds publication-style figures from the JSON (PNG + PDF under `figures_ccfraud/`) |
| `results/ccfraud_results.json` | Latest aggregated metrics (bundled so figures can be rebuilt without a full rerun) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

The experiment needs `creditcard.csv`. Either:

1. **Local file:** download from Kaggle and place `creditcard.csv` in the project root (this path is gitignored), or  
2. **Auto-download:** `pip install kagglehub`, set up [Kaggle API credentials](https://www.kaggle.com/docs/api), then run `experiment_ccfraud.py` (it will fetch the dataset if the CSV is missing).

## Run

**Full experiment** (training can take on the order of ~10+ minutes depending on hardware):

```bash
python3 experiment_ccfraud.py
```

**Figures only** (uses existing `results/ccfraud_results.json`):

```bash
python3 viz_ccfraud.py
```

Generated plots are written to `figures_ccfraud/`. That directory is listed in `.gitignore` so images are not committed by default; rerun `viz_ccfraud.py` after cloning if you want them locally.

## Requirements

See `requirements.txt` for pinned minimum versions (NumPy, pandas, SciPy, scikit-learn, XGBoost, matplotlib).

## Author

Archana Verma — [archverma24/research](https://github.com/archverma24/research)
