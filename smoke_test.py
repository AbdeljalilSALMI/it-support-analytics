from importlib import import_module

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    packages = (
        "numpy",
        "matplotlib",
        "seaborn",
        "joblib",
        "shap",
        "psycopg",
        "streamlit",
        "jupyterlab",
        "pytest",
    )
    for package in packages:
        import_module(package)

    tickets = pd.DataFrame({"category": ["access", "network", "access", "network"]})
    labels = [1, 0, 1, 0]
    pipeline = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"), LogisticRegression()
    )
    pipeline.fit(tickets, labels)
    prediction = pipeline.predict(pd.DataFrame({"category": ["access"]}))
    if prediction.tolist() != [1]:
        raise RuntimeError("Unexpected prediction for the synthetic access ticket.")

    print("Environment check passed: imports and sample prediction succeeded.")


if __name__ == "__main__":
    main()
