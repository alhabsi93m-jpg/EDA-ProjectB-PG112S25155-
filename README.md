# EDA Project B - Time-Series Forecasting Starter

Student: Marwa  
Student ID: PG112S25155

This repository contains a starter Streamlit app for Mini Project B. The app loads a cleaned sample dataset, audits it, prepares baseline time-series features, exports submission files, and includes the fixed AI grader prompt.

## Files

- `app.py` - one-file Streamlit app
- `requirements.txt` - Python dependencies
- `data/dataset_sample.csv` - cleaned dataset slice

## Dataset choices

- Timestamp column: `DATE_TIME`
- Target column: `IRRADIATION`
- Rows included: 3,182

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the full `data` folder.
3. Go to Streamlit Community Cloud.
4. Choose **New app**.
5. Connect your GitHub repository.
6. Set the branch to `main`.
7. Set the main file path to `app.py`.
8. Click **Deploy**.

## OpenRouter API key

The app does not hardcode any API key. For the AI grader, provide the key using one of these methods:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field in the app sidebar

## What to submit

Submit:

- Streamlit deployed app URL
- GitHub repository URL
- `submission.json` exported from the app
- `project_card.md` exported from the app
- Required screenshots:
  - first 10 rows preview
  - metrics table after you add modeling
  - at least one dashboard plot
