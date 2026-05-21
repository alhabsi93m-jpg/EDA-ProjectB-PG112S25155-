import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = '''# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
'''


st.set_page_config(page_title="Mini Project B - Time-Series Forecasting", layout="wide")


def get_openrouter_api_key():
    """Read the OpenRouter key without hardcoding it."""
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass

    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key

    return st.sidebar.text_input(
        "OpenRouter API key",
        type="password",
        help="Used only when you press the AI grader button. It is not saved in this app.",
    )


def audit_dataframe(dataframe):
    audit = pd.DataFrame({
        "column": dataframe.columns,
        "dtype": [str(dataframe[col].dtype) for col in dataframe.columns],
        "missing_percent": [float(dataframe[col].isna().mean() * 100) for col in dataframe.columns],
        "unique_count": [int(dataframe[col].nunique(dropna=True)) for col in dataframe.columns],
    })
    return audit


def clean_time_series(dataframe, timestamp_column, target_column):
    cleaned = dataframe.copy()
    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")
    cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
    cleaned = cleaned.dropna(subset=[timestamp_column, target_column])
    cleaned = cleaned.sort_values(timestamp_column).reset_index(drop=True)
    return cleaned


def resample_time_series(dataframe, timestamp_column, target_column, rule):
    if rule == "No resampling":
        return dataframe

    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    if target_column not in numeric_cols:
        numeric_cols.append(target_column)

    resampled = (
        dataframe.set_index(timestamp_column)[numeric_cols]
        .resample(rule)
        .mean()
        .reset_index()
        .dropna(subset=[target_column])
    )
    return resampled


def build_baseline_features(dataframe, timestamp_column, target_column, horizon):
    feature_df = dataframe[[timestamp_column, target_column]].copy()
    feature_df = feature_df.sort_values(timestamp_column).reset_index(drop=True)

    feature_df["lag_1"] = feature_df[target_column].shift(1)
    feature_df["lag_24"] = feature_df[target_column].shift(24)
    feature_df["rolling_mean_24"] = feature_df[target_column].shift(1).rolling(window=24, min_periods=24).mean()

    feature_df["hour"] = feature_df[timestamp_column].dt.hour
    feature_df["weekend"] = feature_df[timestamp_column].dt.dayofweek.isin([5, 6]).astype(int)
    feature_df["month"] = feature_df[timestamp_column].dt.month

    feature_df["y_target"] = feature_df[target_column].shift(-horizon)

    feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    model_table = feature_df.dropna(subset=feature_cols + ["y_target"]).reset_index(drop=True)
    X = model_table[feature_cols]
    y = model_table["y_target"]
    return model_table, X, y, feature_cols


def make_json_safe(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def parse_grader_response(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def call_openrouter(api_key, evidence_json):
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", evidence_json)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "UTAS EDA Mini Project B",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


st.title("Mini Project B Starter - Time-Series Forecasting")
st.caption("Starter app for UTAS Energy Data Analytics. Students must add modeling, evaluation, and dashboard improvements.")

with st.sidebar:
    st.header("Student information")
    student_name = st.text_input("Student name", value="Marwa")
    student_id = st.text_input("Student ID", value="PG112S25155")
    deployed_url = st.text_input("Streamlit deployed URL")
    project_title = st.text_input("Project title", value="Solar Irradiation Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast the target time-series variable using timestamp-based baseline features, then improve the project with student-added models, metrics, and visuals.",
        height=120,
    )
    api_key = get_openrouter_api_key()

st.header("1. Load local dataset")
default_path = "data/dataset_sample.csv"
dataset_path = st.text_input("Dataset path", value=default_path)

try:
    df = pd.read_csv(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset from {dataset_path}: {exc}")
    st.stop()

st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset audit")
audit = audit_dataframe(df)
st.dataframe(audit, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Missing percentage")
    st.dataframe(audit[["column", "missing_percent"]].sort_values("missing_percent", ascending=False).head(10), use_container_width=True)
with right:
    st.subheader("Column types")
    st.dataframe(audit[["column", "dtype", "unique_count"]], use_container_width=True)

st.header("2. Choose timestamp and target")
timestamp_options = list(df.columns)
target_options = list(df.columns)

default_timestamp_index = timestamp_options.index("DATE_TIME") if "DATE_TIME" in timestamp_options else 0
default_target_index = target_options.index("IRRADIATION") if "IRRADIATION" in target_options else 0

timestamp_col = st.selectbox("Timestamp column", timestamp_options, index=default_timestamp_index)
target_col = st.selectbox("Target column", target_options, index=default_target_index)

cleaned = clean_time_series(df, timestamp_col, target_col)

if cleaned.empty:
    st.error("No valid rows remain after timestamp parsing and target conversion. Choose different columns or fix the dataset.")
    st.stop()

st.success(f"Cleaned rows available: {len(cleaned):,}")

coverage = {
    "start_time": cleaned[timestamp_col].min(),
    "end_time": cleaned[timestamp_col].max(),
    "rows": len(cleaned),
    "target_missing_after_cleaning": int(cleaned[target_col].isna().sum()),
}
st.write("Time coverage", {k: make_json_safe(v) for k, v in coverage.items()})

st.header("3. Optional resampling and forecast horizon")
resampling_choice = st.selectbox(
    "Resampling",
    ["No resampling", "15min", "30min", "1H", "1D"],
    index=0,
    help="Use resampling if the dataset has irregular timestamps or you want a simpler forecasting interval.",
)
horizon = st.number_input("Forecast horizon in rows after optional resampling", min_value=1, max_value=168, value=1, step=1)

prepared = resample_time_series(cleaned, timestamp_col, target_col, resampling_choice)
prepared = prepared.dropna(subset=[target_col]).reset_index(drop=True)

if len(prepared) < 30:
    st.error("Not enough rows remain after cleaning/resampling to create baseline features.")
    st.stop()

st.write(f"Prepared dataset rows: {len(prepared):,}")

st.header("4. Baseline feature table")
feature_table, X, y, feature_cols = build_baseline_features(prepared, timestamp_col, target_col, int(horizon))

if feature_table.empty:
    st.error("Feature table is empty. Try a smaller horizon or avoid aggressive resampling.")
    st.stop()

st.write(f"Feature rows: {len(feature_table):,}")
st.write("Feature columns:", feature_cols)
st.dataframe(feature_table.head(20), use_container_width=True)

st.subheader("Prepared X and y")
col_x, col_y = st.columns(2)
with col_x:
    st.write("X preview")
    st.dataframe(X.head(10), use_container_width=True)
with col_y:
    st.write("y preview")
    st.dataframe(y.head(10).to_frame(name="y_target"), use_container_width=True)

st.header("5. STUDENT ADDITIONS - MODELING")
st.info("Add your forecasting models and evaluation metrics below this marker. Create a pandas DataFrame named results_df.")
st.code(
    """
# STUDENT ADDITIONS - MODELING
# Paste your model training, time-based split, predictions, and metrics here.
# Required evidence for a stronger grade:
# - time-based train/test split
# - at least two models or a clear comparison
# - metrics table with MAE, RMSE, and/or MAPE
# - short explanation of results

results_df = None
""",
    language="python",
)

results_df = None

st.header("6. STUDENT ADDITIONS - DASHBOARD")
st.info("Add extra plots, KPIs, forecast comparisons, and written insights below this marker.")
st.code(
    """
# STUDENT ADDITIONS - DASHBOARD
# Paste additional visuals and insights here.
# Suggested additions:
# - target over time plot
# - actual vs predicted plot after you create predictions
# - KPI cards for metrics
# - comments about missing timestamps, outliers, and resampling
""",
    language="python",
)

with st.expander("Starter target plot"):
    plot_df = prepared[[timestamp_col, target_col]].tail(500)
    fig, ax = plt.subplots()
    ax.plot(plot_df[timestamp_col], plot_df[target_col])
    ax.set_xlabel(timestamp_col)
    ax.set_ylabel(target_col)
    ax.set_title(f"Recent {target_col} values")
    fig.autofmt_xdate()
    st.pyplot(fig)

st.header("7. Export submission files")

has_metrics_table = isinstance(results_df, pd.DataFrame)
results_table = [] if results_df is None else results_df.to_dict(orient="records")

submission = {
    "student_name": student_name,
    "student_id": student_id,
    "deployed_url": deployed_url,
    "project_title": project_title,
    "project_goal": project_goal,
    "dataset_path": dataset_path,
    "timestamp_column": timestamp_col,
    "target_column": target_col,
    "resampling": resampling_choice,
    "forecast_horizon": int(horizon),
    "cleaned_rows": int(len(cleaned)),
    "prepared_rows": int(len(prepared)),
    "feature_rows": int(len(feature_table)),
    "feature_columns": feature_cols,
    "time_coverage": {k: make_json_safe(v) for k, v in coverage.items()},
    "evidence_flags": {
        "has_dataset_preview": True,
        "has_dataset_audit": True,
        "has_timestamp_selection": bool(timestamp_col),
        "has_target_selection": bool(target_col),
        "has_baseline_features": True,
        "has_metrics_table": has_metrics_table,
        "has_student_modeling_additions": False,
        "has_student_dashboard_additions": False,
        "discusses_missing_timestamps": False,
        "discusses_outliers": False,
        "discusses_resampling": resampling_choice != "No resampling",
        "has_insights": False,
    },
    "results_table": results_table,
}

submission_json = json.dumps(submission, indent=2, default=make_json_safe)

project_card = f"""# {project_title}

## Student
- Name: {student_name}
- ID: {student_id}

## Goal
{project_goal}

## Dataset
- Path: {dataset_path}
- Timestamp column: {timestamp_col}
- Target column: {target_col}
- Cleaned rows: {len(cleaned):,}
- Prepared rows: {len(prepared):,}

## Feature preparation
Baseline features prepared:
{", ".join(feature_cols)}

## Student additions needed
Add modeling, evaluation metrics, dashboard improvements, and written insights before final submission.

## Submission links
- Streamlit app URL: {deployed_url}
"""

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col2:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

st.header("8. AI grader /80")
st.warning("Run this only after you add your own modeling, metrics, dashboard, and insights. The starter alone will score low.")

if st.button("Run AI grader"):
    if not api_key:
        st.error("Enter an OpenRouter API key in the sidebar, Streamlit Secrets, or environment variable.")
    else:
        try:
            with st.spinner("Calling AI grader..."):
                raw_output = call_openrouter(api_key, submission_json)
            parsed = parse_grader_response(raw_output)
            if parsed is not None:
                st.subheader("Parsed grader JSON")
                st.json(parsed)
            else:
                st.subheader("Raw grader output")
                st.text(raw_output)
        except Exception as exc:
            st.error(f"AI grader failed: {exc}")
