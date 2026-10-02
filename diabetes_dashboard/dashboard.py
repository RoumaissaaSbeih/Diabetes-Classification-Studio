from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .config import FEATURES, METRICS_PATH, MODEL_PATH, TARGET_COLUMN, ZERO_AS_MISSING
from .data import load_dataset
from .modeling import load_metrics, predict_patient, train_and_save_artifacts

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

.stApp {
  background:
    radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 32%),
    radial-gradient(circle at top right, rgba(224, 122, 95, 0.18), transparent 24%),
    linear-gradient(180deg, #f9fcfb 0%, #eef6f4 100%);
}

.block-container {
  padding-top: 2.2rem;
  padding-bottom: 2rem;
}

.hero {
  padding: 1.8rem 2rem;
  border-radius: 26px;
  background: linear-gradient(135deg, rgba(10, 36, 99, 0.96), rgba(15, 118, 110, 0.92));
  color: #f8fffd;
  box-shadow: 0 24px 60px rgba(16, 42, 67, 0.16);
  margin-bottom: 1.25rem;
}

.hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-size: 2.45rem;
  line-height: 1.05;
  margin: 0.35rem 0 0.55rem 0;
}

.hero p,
.hero span {
  font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
}

.eyebrow {
  display: inline-block;
  font-size: 0.82rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.82;
}

.metric-card {
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(16, 42, 67, 0.08);
  border-radius: 20px;
  padding: 1rem 1.1rem;
  box-shadow: 0 16px 38px rgba(16, 42, 67, 0.08);
  min-height: 126px;
}

.metric-label {
  color: #486581;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

.metric-value {
  color: #102a43;
  font-family: "Fraunces", Georgia, serif;
  font-size: 2rem;
  margin-top: 0.3rem;
}

.metric-subtext {
  color: #52606d;
  font-size: 0.9rem;
}

.panel-note {
  padding: 0.9rem 1rem;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(15, 118, 110, 0.12);
  border-radius: 16px;
  margin-bottom: 1rem;
}

.risk-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.85rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.95rem;
}

.risk-low {
  background: rgba(15, 118, 110, 0.14);
  color: #0f766e;
}

.risk-mid {
  background: rgba(180, 83, 9, 0.14);
  color: #b45309;
}

.risk-high {
  background: rgba(190, 24, 93, 0.14);
  color: #be185d;
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _ensure_ready_assets() -> None:
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return
    with st.spinner("Training the diabetes model for the first launch..."):
        train_and_save_artifacts(force=True)


@st.cache_data(show_spinner=False)
def _load_dataframe() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def _load_metrics() -> dict[str, Any]:
    return load_metrics()


def _metric_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_sidebar(metrics: dict[str, Any]) -> None:
    final_model = metrics["final_model"]
    st.sidebar.title("Project Navigator")
    st.sidebar.markdown("A Kaggle-inspired classification workflow, upgraded into a deployable dashboard.")
    st.sidebar.info(
        "Zero values for Glucose, BloodPressure, SkinThickness, Insulin, and BMI are treated as missing and imputed with medians."
    )
    st.sidebar.metric("Deployment Model", final_model["name"])
    st.sidebar.metric("Cross-Validated ROC-AUC", f'{final_model["cv_best_score"]:.3f}')
    st.sidebar.metric("Test F1 Score", f'{final_model["test_metrics"]["f1"]:.3f}')
    st.sidebar.caption("Use the Prediction Studio tab to score new patient profiles live inside the app.")


def _render_overview(frame: pd.DataFrame, metrics: dict[str, Any]) -> None:
    summary = metrics["dataset_summary"]
    model_metrics = metrics["final_model"]["test_metrics"]

    cols = st.columns(4)
    with cols[0]:
        _metric_card("Dataset Rows", f'{summary["rows"]}', "Pima Indians Diabetes observations")
    with cols[1]:
        _metric_card("Positive Rate", f'{summary["positive_rate"]}%', "Observed diabetes outcome share")
    with cols[2]:
        _metric_card("Model Accuracy", f'{model_metrics["accuracy"]:.3f}', "Held-out test split")
    with cols[3]:
        _metric_card("ROC-AUC", f'{model_metrics["roc_auc"]:.3f}', "Probability ranking strength")

    st.markdown(
        """
        <div class="panel-note">
        This dashboard keeps the classic notebook flow intact: explore the data, tune the classifier, evaluate the report,
        and finish with a production-like prediction surface for new patient scenarios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns((1, 1))
    with left:
        outcome_counts = (
            frame[TARGET_COLUMN]
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Patients")
            .replace({"Outcome": {0: "No diabetes", 1: "Diabetes"}})
        )
        fig = px.pie(
            outcome_counts,
            values="Patients",
            names="Outcome",
            hole=0.58,
            color="Outcome",
            color_discrete_map={"No diabetes": "#0f766e", "Diabetes": "#e76f51"},
        )
        fig.update_layout(title="Outcome Balance", margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            frame,
            x="Glucose",
            y="BMI",
            color=frame[TARGET_COLUMN].map({0: "No diabetes", 1: "Diabetes"}),
            size="Age",
            hover_data=["Pregnancies", "BloodPressure", "Insulin"],
            color_discrete_map={"No diabetes": "#0f766e", "Diabetes": "#e76f51"},
            title="Glucose vs BMI by Observed Outcome",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns((1.1, 0.9))
    with left:
        correlation = frame[FEATURES + [TARGET_COLUMN]].corr(numeric_only=True).round(2)
        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Tealgrn",
            title="Feature Correlation Matrix",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            frame.replace({TARGET_COLUMN: {0: "No diabetes", 1: "Diabetes"}}),
            x=TARGET_COLUMN,
            y="Age",
            color=TARGET_COLUMN,
            color_discrete_map={"No diabetes": "#0f766e", "Diabetes": "#e76f51"},
            title="Age Distribution by Outcome",
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(frame.head(12), use_container_width=True, hide_index=True)


def _render_model_lab(metrics: dict[str, Any]) -> None:
    benchmark = pd.DataFrame(metrics["benchmark_results"])
    final_model = metrics["final_model"]

    left, right = st.columns((1, 1))
    with left:
        melted = benchmark.melt(id_vars="model", value_vars=["accuracy", "f1", "roc_auc"], var_name="Metric", value_name="Score")
        fig = px.bar(
            melted,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Benchmark Models on the Test Split",
            color_discrete_sequence=["#0f766e", "#f4a261", "#1d3557"],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), xaxis_title="", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        report = pd.DataFrame(final_model["classification_report"]).transpose()
        report = report.loc[[idx for idx in report.index if idx != "accuracy"]]
        st.subheader("Classification Report")
        st.dataframe(report.round(3), use_container_width=True)
        st.caption(
            f'Best parameters: {final_model["best_params"]} | Cross-validated ROC-AUC: {final_model["cv_best_score"]:.3f}'
        )

    left, right = st.columns((0.85, 1.15))
    with left:
        matrix = final_model["confusion_matrix"]
        fig = px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            color_continuous_scale="Tealgrn",
            title="Confusion Matrix",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        roc_frame = pd.DataFrame(final_model["roc_curve"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=roc_frame["fpr"],
                y=roc_frame["tpr"],
                mode="lines",
                line=dict(color="#0f766e", width=4),
                name="Tuned KNN",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#9fb3c8", dash="dash"),
                name="Random baseline",
            )
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    importance = pd.DataFrame(final_model["feature_importance"])
    fig = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Permutation Importance of the Final Model",
        color="importance",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)


def _default_inputs(metrics: dict[str, Any]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for feature, stats in metrics["feature_ranges"].items():
        defaults[feature] = stats["median"]
    return defaults


def _feature_percentiles(frame: pd.DataFrame, profile: dict[str, float]) -> pd.DataFrame:
    rows = []
    for feature, value in profile.items():
        percentile = float((frame[feature] <= value).mean() * 100)
        rows.append({"Feature": feature, "Input": value, "Percentile": round(percentile, 1)})
    return pd.DataFrame(rows).sort_values("Percentile", ascending=False)


def _risk_class(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "risk-high", "High likelihood pattern"
    if probability >= 0.4:
        return "risk-mid", "Moderate likelihood pattern"
    return "risk-low", "Lower likelihood pattern"


def _render_prediction_studio(frame: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="panel-note">
        Try a patient scenario below. The final tuned KNN model is loaded from the saved artifact, so the web app is using the trained model directly rather than re-creating results manually.
        </div>
        """,
        unsafe_allow_html=True,
    )

    defaults = _default_inputs(metrics)
    input_cols = st.columns(2)
    field_values: dict[str, float] = {}

    with st.form("prediction-form"):
        for index, feature in enumerate(FEATURES):
            stats = metrics["feature_ranges"][feature]
            step = 0.001 if feature == "DiabetesPedigreeFunction" else 0.1
            if feature in {"Pregnancies", "Age", "Glucose", "BloodPressure", "SkinThickness", "Insulin"}:
                step = 1.0

            with input_cols[index % 2]:
                field_values[feature] = st.number_input(
                    feature,
                    min_value=float(stats["min"]),
                    max_value=float(stats["max"]),
                    value=float(defaults[feature]),
                    step=step,
                    help="A value of 0 will be treated as missing for select clinical features."
                    if feature in ZERO_AS_MISSING
                    else None,
                )

        submitted = st.form_submit_button("Predict Diabetes Risk")

    if submitted:
        prediction = predict_patient(field_values)
        st.session_state["latest_prediction"] = {"inputs": field_values, "result": prediction}

    if "latest_prediction" not in st.session_state:
        return

    latest = st.session_state["latest_prediction"]
    result = latest["result"]
    probability = result["probability"]
    percentile_frame = _feature_percentiles(frame, latest["inputs"])
    risk_class, risk_text = _risk_class(probability)

    left, right = st.columns((0.9, 1.1))
    with left:
        st.markdown(
            f'<div class="risk-pill {risk_class}">{risk_text} • {probability * 100:.1f}% predicted probability</div>',
            unsafe_allow_html=True,
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 38, "color": "#102a43"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f766e"},
                    "steps": [
                        {"range": [0, 40], "color": "#d9f2ec"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fde2e4"},
                    ],
                },
                title={"text": "Predicted Risk"},
            )
        )
        gauge.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=320)
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(result["label"])

    with right:
        st.subheader("Profile Context")
        top_percentiles = percentile_frame.head(4)
        fig = px.bar(
            top_percentiles.sort_values("Percentile"),
            x="Percentile",
            y="Feature",
            orientation="h",
            text="Input",
            title="Highest Relative Inputs vs the Training Population",
            color="Percentile",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), yaxis_title="", xaxis_title="Percentile")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(percentile_frame, use_container_width=True, hide_index=True)

    st.warning(
        "This app is an educational classifier demo based on historical data. It estimates risk patterns and is not a clinical diagnosis."
    )


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Diabetes Classification Studio",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _ensure_ready_assets()

    frame = _load_dataframe()
    metrics = _load_metrics()

    st.markdown(
        """
        <div class="hero">
          <span class="eyebrow">Machine Learning Dashboard</span>
          <h1>Diabetes Classification Studio</h1>
          <p>Explore the dataset, inspect the tuned model, and score new patient profiles in a deployment-ready web app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _build_sidebar(metrics)

    overview_tab, model_tab, predict_tab = st.tabs(["Executive Dashboard", "Model Lab", "Prediction Studio"])
    with overview_tab:
        _render_overview(frame, metrics)
    with model_tab:
        _render_model_lab(metrics)
    with predict_tab:
        _render_prediction_studio(frame, metrics)
