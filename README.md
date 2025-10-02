# DS Lab Experiments 7 & 8 Platform

This repository provides a unified, step-by-step demonstration for Experiment 7 (CI/CD) and Experiment 8 (Dashboard, XAI, Fairness) using your datasets.

- Final dataset: `prompt/final_processed_zudio_data.csv`
- Raw dataset (Exp 2 only): `prompt/original_raw_data.csv`

## Quickstart

```bash
# Create venv and install
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# Run Streamlit UI
streamlit run streamlit_app.py

# Run API
uvicorn api.main:app --reload
```

## Docker

```bash
docker build -t ds-lab-app .
docker run -p 8000:8000 -p 8501:8501 ds-lab-app
```

## CI/CD (GitHub Actions)
- Workflow at `.github/workflows/ci.yml` runs lint and tests on push/PR to `main`.

## Notes
- Provide PRAW credentials via env vars for live Reddit scrape: `PRAW_CLIENT_ID`, `PRAW_CLIENT_SECRET`, `PRAW_USER_AGENT`.
- Train a model on the Streamlit "Model Selection" page before using the API `/predict`.
