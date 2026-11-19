"""Streamlit pages for inference and model evaluation."""

import datetime as dt
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import OPENING_CATEGORIES, build_opening_features
from src.prediction import explain_slow_resolution


PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
FIGURES = PROJECT / "reports" / "figures"


@st.cache_resource
def load_classifier(path, modified_at):
    return joblib.load(path)


def resolution_prediction():
    st.header("Resolution Prediction")
    path = MODELS / "resolution_classifier.joblib"
    if not path.is_file():
        st.info("Run the machine learning notebook to generate the model.")
        return
    model = load_classifier(path, path.stat().st_mtime_ns)
    encoder = model.steps[0][1].named_transformers_["category"].named_steps["onehotencoder"]
    with st.form("ticket_prediction"):
        columns = st.columns(3)
        ticket = {
            name: column.selectbox(name.title(), [value for value in options if value != "(Missing)"])
            for name, column, options in zip(OPENING_CATEGORIES, columns, encoder.categories_)
        }
        date_column, time_column = st.columns(2)
        date = date_column.date_input("Creation date")
        time = time_column.time_input("Creation time")
        submitted = st.form_submit_button("Predict resolution", type="primary")
    if not submitted:
        return

    ticket["created_at"] = dt.datetime.combine(date, time)
    features = build_opening_features(pd.DataFrame([ticket]))
    prediction = model.predict(features)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    label = "Fast resolution" if prediction == 1 else "Slow resolution"
    st.metric(label, f"{probabilities[prediction]:.1%}", help="Model probability for the predicted class.")
    st.caption("Fast means under 24 hours. The model was evaluated on simulated tickets.")

    baseline, effects = explain_slow_resolution(model, features)
    contributions = effects.iloc[0].sort_values(key=abs, ascending=False)
    factors = pd.DataFrame({
        "Input": contributions.index.str.replace("_", " ").str.title(),
        "Effect on slow score": [f"{100 * value:+.2f} pp" for value in contributions],
    })
    st.subheader("Factors behind the prediction")
    st.caption(f"Slow-resolution score: {probabilities[0]:.1%} · Model baseline: {baseline[0]:.1%}.")
    st.dataframe(factors, hide_index=True, width="stretch")


def model_insights():
    st.header("Model Insights")
    path = MODELS / "dashboard_metrics.json"
    if not path.is_file():
        st.info("Run the machine learning notebook through Export dashboard metrics.")
        return

    metrics = json.loads(path.read_text(encoding="utf-8"))
    st.subheader("Validation comparison")
    st.dataframe(pd.DataFrame.from_dict(metrics["validation"], orient="index").round(3), width="stretch")
    st.subheader("Held-out evaluation")
    st.caption(f"{metrics['selected_model']} · {metrics['test_tickets']:,} test tickets · Fast is the positive class.")
    st.dataframe(pd.DataFrame.from_dict(metrics["test"], orient="index").round(3), width="stretch")

    columns = st.columns(2)
    charts = [
        ("test_confusion_matrix.png", "Test confusion matrix"),
        ("shap_feature_importance.png", "Global SHAP importance"),
    ]
    for column, (name, title) in zip(columns, charts):
        chart = FIGURES / name
        if chart.is_file():
            column.image(str(chart), caption=title, width="stretch")

    example = FIGURES / "shap_slow_resolution_example.png"
    if example.is_file():
        with st.expander("Individual SHAP example"):
            st.image(str(example), width="stretch")
