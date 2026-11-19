"""Explain slow-resolution predictions from the fitted classifier pipeline."""

import numpy as np
import pandas as pd
import shap

from src.model import MODEL_FEATURES


def explain_slow_resolution(pipeline, features):
    """Return baseline and grouped probability contributions for each ticket."""
    preprocessor, classifier = (step for _, step in pipeline.steps)
    encoded = preprocessor.transform(features)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    slow_index = list(classifier.classes_).index(0)
    explanations = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")(encoded)
    names = preprocessor.get_feature_names_out()
    values = explanations.values[:, :, slow_index]
    grouped = {}
    for feature in MODEL_FEATURES:
        mask = [
            name == f"time__{feature}" or name.startswith(f"category__{feature}_")
            for name in names
        ]
        grouped[feature] = values[:, mask].sum(axis=1)
    contributions = pd.DataFrame(grouped, index=features.index)
    baseline = explanations.base_values[:, slow_index]
    np.testing.assert_allclose(
        baseline + contributions.sum(axis=1),
        pipeline.predict_proba(features)[:, slow_index],
        atol=1e-6,
    )
    return baseline, contributions
