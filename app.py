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


def infer_time_gap_summary(dataframe, timestamp_column):
    if len(dataframe) < 3:
        return {
            "median_gap_minutes": None,
            "largest_gap_minutes": None,
            "suspected_missing_intervals": 0,
            "duplicate_timestamps": int(dataframe[timestamp_column].duplicated().sum()),
        }

    sorted_time = dataframe[timestamp_column].sort_values()
    gaps = sorted_time.diff().dropna()
    median_gap = gaps.median()

    if pd.isna(median_gap) or median_gap == pd.Timedelta(0):
        suspected_missing = 0
    else:
        suspected_missing = int((gaps > median_gap * 1.5).sum())

    return {
        "median_gap_minutes": float(median_gap.total_seconds() / 60) if pd.notna(median_gap) else None,
        "largest_gap_minutes": float(gaps.max().total_seconds() / 60) if len(gaps) else None,
        "suspected_missing_intervals": suspected_missing,
        "duplicate_timestamps": int(dataframe[timestamp_column].duplicated().sum()),
    }


def target_outlier_summary(dataframe, target_column):
    series = pd.to_numeric(dataframe[target_column], errors="coerce").dropna()

    if series.empty:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percent": 0.0,
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()

    return {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_count": int(outliers),
        "outlier_percent": float(outliers / len(series) * 100),
    }


def resample_time_series(dataframe, timestamp_column, target_column, rule):
    if rule == "No resampling":
        return dataframe.copy()

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


def add_cyclical_feature(dataframe, source_column, period, prefix):
    dataframe[f"{prefix}_sin"] = np.sin(2 * np.pi * dataframe[source_column] / period)
    dataframe[f"{prefix}_cos"] = np.cos(2 * np.pi * dataframe[source_column] / period)
    return dataframe


def build_improved_features(dataframe, timestamp_column, target_column, horizon):
    feature_df = dataframe.copy()
    feature_df = feature_df.sort_values(timestamp_column).reset_index(drop=True)
    feature_df[target_column] = pd.to_numeric(feature_df[target_column], errors="coerce")

    lag_candidates = [1, 2, 4, 8, 24, 48, 96]
    rolling_windows = [4, 8, 24, 48, 96]

    feature_cols = []

    for lag in lag_candidates:
        if len(feature_df) > lag:
            col_name = f"lag_{lag}"
            feature_df[col_name] = feature_df[target_column].shift(lag)
            feature_cols.append(col_name)

    shifted_target = feature_df[target_column].shift(1)

    for window in rolling_windows:
        if len(feature_df) > window:
            mean_col = f"rolling_mean_{window}"
            std_col = f"rolling_std_{window}"
            min_col = f"rolling_min_{window}"
            max_col = f"rolling_max_{window}"

            feature_df[mean_col] = shifted_target.rolling(window=window, min_periods=max(2, window // 2)).mean()
            feature_df[std_col] = shifted_target.rolling(window=window, min_periods=max(2, window // 2)).std()
            feature_df[min_col] = shifted_target.rolling(window=window, min_periods=max(2, window // 2)).min()
            feature_df[max_col] = shifted_target.rolling(window=window, min_periods=max(2, window // 2)).max()
            feature_cols.extend([mean_col, std_col, min_col, max_col])

    feature_df["diff_1"] = feature_df[target_column].diff(1).shift(1)
    feature_df["diff_4"] = feature_df[target_column].diff(4).shift(1)
    feature_df["target_expanding_mean"] = shifted_target.expanding(min_periods=10).mean()
    feature_cols.extend(["diff_1", "diff_4", "target_expanding_mean"])

    feature_df["hour"] = feature_df[timestamp_column].dt.hour
    feature_df["dayofweek"] = feature_df[timestamp_column].dt.dayofweek
    feature_df["weekend"] = feature_df["dayofweek"].isin([5, 6]).astype(int)
    feature_df["month"] = feature_df[timestamp_column].dt.month
    feature_df["dayofyear"] = feature_df[timestamp_column].dt.dayofyear
    feature_df["is_daylight_hour"] = feature_df["hour"].between(6, 18).astype(int)

    feature_df = add_cyclical_feature(feature_df, "hour", 24, "hour")
    feature_df = add_cyclical_feature(feature_df, "dayofweek", 7, "dayofweek")
    feature_df = add_cyclical_feature(feature_df, "month", 12, "month")
    feature_df = add_cyclical_feature(feature_df, "dayofyear", 365.25, "dayofyear")

    time_features = [
        "hour",
        "dayofweek",
        "weekend",
        "month",
        "dayofyear",
        "is_daylight_hour",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]
    feature_cols.extend(time_features)

    extra_numeric_cols = [
        col for col in feature_df.select_dtypes(include=[np.number]).columns
        if col not in feature_cols + [target_column, "y_target"]
    ]

    for col in extra_numeric_cols:
        if col.upper() in ["PLANT_ID"] or "ID" in col.upper():
            continue
        shifted_col = f"{col}_lag_1"
        feature_df[shifted_col] = feature_df[col].shift(1)
        feature_cols.append(shifted_col)

    feature_df["y_target"] = feature_df[target_column].shift(-int(horizon))

    required_cols = feature_cols + ["y_target"]
    model_table = feature_df.dropna(subset=required_cols).reset_index(drop=True)

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
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
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


st.title("Mini Project B - Time-Series Forecasting")
st.caption("Starter app for UTAS Energy Data Analytics. This version improves feature engineering but still leaves modeling and metrics for the student.")

with st.sidebar:
    st.header("Student information")
    student_name = st.text_input("Student name", value="Marwa")
    student_id = st.text_input("Student ID", value="PG112S25155")
    deployed_url = st.text_input("Streamlit deployed URL")
    project_title = st.text_input("Project title", value="Solar Irradiation Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast solar irradiation using time-series features, rolling statistics, lag features, calendar features, and student-added models and metrics.",
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
    st.dataframe(
        audit[["column", "missing_percent"]]
        .sort_values("missing_percent", ascending=False)
        .head(10),
        use_container_width=True,
    )

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

gap_summary = infer_time_gap_summary(cleaned, timestamp_col)
outlier_summary = target_outlier_summary(cleaned, target_col)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cleaned rows", f"{len(cleaned):,}")
c2.metric("Median gap minutes", "NA" if gap_summary["median_gap_minutes"] is None else f"{gap_summary['median_gap_minutes']:.1f}")
c3.metric("Possible missing intervals", gap_summary["suspected_missing_intervals"])
c4.metric("Target outliers %", f"{outlier_summary['outlier_percent']:.2f}%")

with st.expander("Time coverage, missing timestamp check, and outlier check"):
    st.write("Time coverage", {k: make_json_safe(v) for k, v in coverage.items()})
    st.write("Timestamp gap summary", gap_summary)
    st.write("Target outlier summary", outlier_summary)
    st.info(
        "Evidence note: suspected missing intervals are identified from unusually large timestamp gaps. "
        "Outliers use the IQR rule on the selected target column."
    )

st.header("3. Optional resampling and forecast horizon")
resampling_choice = st.selectbox(
    "Resampling",
    ["No resampling", "15min", "30min", "1H", "1D"],
    index=0,
    help="Use resampling if the dataset has irregular timestamps or you want a simpler forecasting interval.",
)

horizon = st.number_input(
    "Forecast horizon in rows after optional resampling",
    min_value=1,
    max_value=168,
    value=1,
    step=1,
)

prepared = resample_time_series(cleaned, timestamp_col, target_col, resampling_choice)
prepared = prepared.dropna(subset=[target_col]).reset_index(drop=True)

if len(prepared) < 40:
    st.error("Not enough rows remain after cleaning/resampling to create improved features.")
    st.stop()

st.write(f"Prepared dataset rows: {len(prepared):,}")

st.header("4. Improved feature engineering table")
feature_table, X, y, feature_cols = build_improved_features(prepared, timestamp_col, target_col, int(horizon))

if feature_table.empty:
    st.error("Feature table is empty. Try a smaller horizon or avoid aggressive resampling.")
    st.stop()

st.success(f"Feature table created with {len(feature_table):,} rows and {len(feature_cols)} feature columns.")

feature_groups = {
    "Lag features": [col for col in feature_cols if col.startswith("lag_")],
    "Rolling statistics": [col for col in feature_cols if col.startswith("rolling_")],
    "Change/trend features": [col for col in feature_cols if col.startswith("diff_") or col == "target_expanding_mean"],
    "Calendar/cyclical features": [
        col for col in feature_cols
        if col in [
            "hour", "dayofweek", "weekend", "month", "dayofyear", "is_daylight_hour",
            "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
            "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
        ]
    ],
    "Shifted external numeric features": [
        col for col in feature_cols
        if col.endswith("_lag_1") and not col.startswith("lag_")
    ],
}

group_df = pd.DataFrame([
    {"feature_group": group, "count": len(cols), "features": ", ".join(cols)}
    for group, cols in feature_groups.items()
])
st.subheader("Feature groups")
st.dataframe(group_df, use_container_width=True)

st.subheader("Feature table preview")
st.dataframe(feature_table[[timestamp_col, target_col] + feature_cols + ["y_target"]].head(20), use_container_width=True)

st.subheader("Prepared X and y")
col_x, col_y = st.columns(2)
with col_x:
    st.write("X preview")
    st.dataframe(X.head(10), use_container_width=True)
with col_y:
    st.write("y preview")
    st.dataframe(y.head(10).to_frame(name="y_target"), use_container_width=True)

st.header("5. STUDENT ADDITIONS - MODELING")
st.info("Paste your model training, time-based split, predictions, and metrics below this marker. Create a pandas DataFrame named results_df.")

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
st.info("Paste additional plots, KPIs, forecast comparisons, and written insights below this marker.")

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

st.subheader("Starter dashboard visuals")
with st.expander("Target over time"):
    plot_df = prepared[[timestamp_col, target_col]].tail(500)
    fig, ax = plt.subplots()
    ax.plot(plot_df[timestamp_col], plot_df[target_col])
    ax.set_xlabel(timestamp_col)
    ax.set_ylabel(target_col)
    ax.set_title(f"Recent {target_col} values")
    fig.autofmt_xdate()
    st.pyplot(fig)

with st.expander("Target distribution"):
    fig, ax = plt.subplots()
    ax.hist(prepared[target_col].dropna(), bins=30)
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {target_col}")
    st.pyplot(fig)

with st.expander("Average target by hour"):
    hourly_profile = prepared.copy()
    hourly_profile["hour"] = hourly_profile[timestamp_col].dt.hour
    hourly_profile = hourly_profile.groupby("hour", as_index=False)[target_col].mean()

    fig, ax = plt.subplots()
    ax.plot(hourly_profile["hour"], hourly_profile[target_col], marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(f"Average {target_col}")
    ax.set_title(f"Average {target_col} by hour")
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
    "feature_groups": feature_groups,
    "time_coverage": {k: make_json_safe(v) for k, v in coverage.items()},
    "timestamp_gap_summary": gap_summary,
    "target_outlier_summary": outlier_summary,
    "evidence_flags": {
        "has_dataset_preview": True,
        "has_dataset_audit": True,
        "has_timestamp_selection": bool(timestamp_col),
        "has_target_selection": bool(target_col),
        "has_baseline_features": True,
        "has_improved_feature_engineering": True,
        "has_lag_features": len(feature_groups["Lag features"]) > 0,
        "has_rolling_features": len(feature_groups["Rolling statistics"]) > 0,
        "has_cyclical_time_features": len(feature_groups["Calendar/cyclical features"]) > 0,
        "has_external_numeric_features": len(feature_groups["Shifted external numeric features"]) > 0,
        "has_metrics_table": has_metrics_table,
        "has_student_modeling_additions": False,
        "has_student_dashboard_additions": False,
        "discusses_missing_timestamps": True,
        "discusses_outliers": True,
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

## Time-series integrity checks
- Median timestamp gap in minutes: {gap_summary["median_gap_minutes"]}
- Largest timestamp gap in minutes: {gap_summary["largest_gap_minutes"]}
- Suspected missing timestamp intervals: {gap_summary["suspected_missing_intervals"]}
- Duplicate timestamps: {gap_summary["duplicate_timestamps"]}
- Target outlier percent by IQR rule: {outlier_summary["outlier_percent"]:.2f}%

## Improved feature preparation
Feature groups:
{group_df.to_markdown(index=False)}

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
