"""Chronological evaluation splits and model preprocessing."""

from time import perf_counter

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import OPENING_CATEGORIES, TARGET


TIME_FEATURES = ("hour_created", "day_of_week", "month_created", "is_weekend")
MODEL_FEATURES = (*OPENING_CATEGORIES, *TIME_FEATURES)


def chronological_splits(labeled, outcomes, validation_start, test_start):
    """Keep training and validation labels known before the next period starts."""
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("Validation must begin before the test period.")

    rows = labeled.merge(
        outcomes[["ticket_id", "resolved_at"]], on="ticket_id", how="left", validate="one_to_one"
    )
    if rows[["created_at", "resolved_at", TARGET]].isna().any().any():
        raise ValueError("Labeled tickets require creation and resolution timestamps and a target.")
    rows = rows.sort_values(["created_at", "ticket_id"])
    created = rows["created_at"]
    resolved = rows["resolved_at"]
    masks = {
        "train": created.lt(validation_start) & resolved.lt(validation_start),
        "validation": created.ge(validation_start) & created.lt(test_start) & resolved.lt(test_start),
        "test": created.ge(test_start),
    }
    return {name: rows.loc[mask].copy() for name, mask in masks.items()}


def numeric_array(frame):
    return frame.apply(pd.to_numeric).to_numpy(dtype=float, na_value=float("nan"))


def make_preprocessor():
    """Return unfitted transformations for a scikit-learn model pipeline."""
    categories = make_pipeline(
        SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value="(Missing)"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    time = make_pipeline(
        FunctionTransformer(numeric_array, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    return ColumnTransformer([
        ("category", categories, list(OPENING_CATEGORIES)),
        ("time", time, list(TIME_FEATURES)),
    ])


def make_candidates():
    """Give each classifier its own unfitted preprocessing pipeline."""
    classifiers = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=500),
        "Decision tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=20, random_state=42,
        ),
    }
    return {name: make_pipeline(make_preprocessor(), model) for name, model in classifiers.items()}


def validation_scores(model, features, target):
    """Report classification metrics with fast resolution as the positive class."""
    predictions = model.predict(features)
    fast_probabilities = model.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(target, fast_probabilities),
    }


def train_candidates(train_features, train_target, validation_features, validation_target):
    models = make_candidates()
    scores = {}
    for name, model in models.items():
        started = perf_counter()
        model.fit(train_features, train_target)
        fit_seconds = perf_counter() - started
        scores[name] = validation_scores(model, validation_features, validation_target)
        scores[name]["Fit seconds"] = fit_seconds
    return models, pd.DataFrame.from_dict(scores, orient="index")
