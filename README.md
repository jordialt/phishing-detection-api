# Phishing Site Detector

TL;DR

A small project that downloads public phishing datasets, processes URL-based features, trains multiple classifiers (Logistic Regression, Random Forest, XGBoost) using nested cross-validation to avoid overfitting, and exposes a Flask API + demo UI to test URL predictions. The repo includes scripts for diagnosis and feature inspection to detect leakage.

Why this project
- Real-world problem: phishing detection using only URL-derived features.
- Focus on reproducibility and evaluation rigor (nested CV).
- Includes an API and simple UI to demo predictions.

Repo structure

- `app.py` — Flask API and demo UI (endpoints `/predict` and `/predict_proba`).
- `src/download_data.py` — download scripts (OpenPhish and Kaggle) and combine datasets.
- `src/process_data.py` — data cleaning and feature extraction.
- `src/train_model.py` — training + nested CV and model comparison (saves best models).
- `src/diagnose_leakage.py` — quick checks for duplicates, ID-like columns, and perfect predictors.
- `src/feature_inspect.py` — per-feature single-feature predictive power and distributions.
- `scripts/debug_url.py` — debug a single URL against saved models (prints features, preds, probs, importances).
- `data/` — raw and intermediate datasets (kept small in repo).
- `archive/` — old models and plots moved here during cleanup.

Quick start

1. Create a virtual environment and install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download data (requires Kaggle credentials for Kaggle downloads):

```bash
python src/download_data.py
```

3. Process data:

```bash
python src/process_data.py
```

4. (Option A) Fast train (reduced nested CV for demo):

```bash
python src/train_model.py
```

5. Run the API locally and open the demo:

```bash
python app.py
# Open http://localhost:5000/
```

Reproducibility notes
- Models trained with nested CV; hyperparameter grids are intentionally small for demo speed.
- A diagnostic script `src/diagnose_leakage.py` and `src/feature_inspect.py` help detect features that leak the label.
- The training script also produces `processed_data_cleaned.csv` showing features dropped by the leakage heuristic.

Results
- See `models_comparison.csv` for a table of nested CV and test metrics per model.
- Confusion matrices and F1 plots are in the repo root (or `archive/` if moved).

Caveats
- Some features (e.g., raw `url`) can lead to data leakage if included as unique identifiers. I detected and removed ID-like columns before retraining.
- Nested CV is computationally expensive; use the `--min-count` or reduce folds to speed up.

Future work
- Replace manual feature engineering with a robust sklearn Pipeline that includes safe URL vectorization and preserves preprocessing during serialization.
- Add more diverse legitimate sites and adversarial examples.
- Deploy as serverless function or container with authentication and rate limits.

License & attribution
- Dataset: `phishing-site-urls` from Kaggle (link in `src/download_data.py`). Check dataset license on Kaggle.

Contact
- Repo owner: jordi
- Demo: http://localhost:5000/
