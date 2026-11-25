# AI-Based Cardiovascular Disease Prediction
Comparative ML project: **Decision Tree · Naive Bayes · KNN · ANN (MLP)** on the 70,000-record `cardio_train.csv` dataset, with a Streamlit app.

## Folder structure
```
cvd_project/
├── app.py                         # Streamlit app (predict, compare, importance, system design)
├── train.py                       # trains/tunes/evaluates all 4 models, saves models + metrics + figures
├── src_common.py                  # shared cleaning + feature engineering (used by train.py and app.py)
├── cardio_ml_comparison.ipynb     # full analysis notebook (EDA, CV, evaluation, interpretation)
├── data/cardio_train.csv
├── models/                        # *.joblib + metrics.json (created by train.py)
├── reports/figures/               # PNG figures
├── requirements.txt
└── .vscode/ (launch + settings), .streamlit/config.toml
```

## Run in VS Code
1. Unzip, then **File → Open Folder…** → `cvd_project`.
2. Terminal → New Terminal:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py            # a few minutes; creates models/ and reports/
streamlit run app.py       # opens http://localhost:8501
```
3. Notebook: open `cardio_ml_comparison.ipynb`, select the `.venv` kernel, **Run All**.
   (Or use Run and Debug panel: "Train models" / "Streamlit app".)

## Deploy on Streamlit Community Cloud
1. Push the whole folder to a GitHub repo (**include `models/`** – the app loads the trained files; `.gitignore` does not exclude it).
2. Go to share.streamlit.io → New app → pick the repo, branch `main`, main file `app.py` → Deploy.
3. Keep `requirements.txt` in the repo root. If you retrain locally with a different scikit-learn version, re-commit `models/`, and pin the same version in `requirements.txt`.

## Notes
* Cleaning removes duplicates and impossible values (e.g. systolic BP > 250, height < 120 cm, systolic ≤ diastolic).
* Scaling lives inside each sklearn `Pipeline` (fit on training folds only → no leakage).
* Educational prototype – **not a medical device**.
