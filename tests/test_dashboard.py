from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from streamlit.testing.v1 import AppTest


PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "app" / "app.py"


def test_dashboard_handles_database_connection_error():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    with patch("psycopg.connect", side_effect=psycopg.OperationalError("test connection error")):
        app.button[0].click().run()
    assert not app.exception
    assert "Could not load tickets" in app.error[0].value


def test_model_pages_work_without_postgres():
    artifacts = PROJECT / "models"
    if not all((artifacts / name).is_file() for name in (
        "resolution_classifier.joblib", "dashboard_metrics.json",
    )):
        pytest.skip("Generate the model artifacts with the machine learning notebook.")
    with patch("psycopg.connect", side_effect=AssertionError("Unexpected DB connection")):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.radio[0].set_value("Resolution Prediction").run()
        app.button[0].click().run()
        assert not app.exception
        assert app.metric[0].label in {"Fast resolution", "Slow resolution"}
        assert len(app.dataframe[0].value) == 7

        app.radio[0].set_value("Model Insights").run()
        assert not app.exception
        assert len(app.dataframe) == 2
        assert "ROC-AUC" in app.dataframe[1].value


def test_ticket_analytics_filters_database_rows():
    connection = MagicMock()
    cursor = connection.__enter__.return_value.execute.return_value
    columns = ("created_at", "category", "priority", "assigned_team", "channel",
               "resolution_wallclock_hours", "is_fast_resolution")
    cursor.description = [SimpleNamespace(name=column) for column in columns]
    cursor.fetchall.return_value = [
        (datetime(2024, 1, 1), "Billing", "High", "Tier 1", "Email", 2.0, True),
        (datetime(2024, 1, 2), "Refunds", "Low", "Tier 1", "Phone", None, None),
    ]
    with patch("psycopg.connect", return_value=connection):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button[0].click().run()
    assert not app.exception and app.metric[0].value == "2"

    app.radio[0].set_value("Ticket Analytics").run()
    next(widget for widget in app.multiselect if widget.label == "Category").set_value(["Refunds"]).run()
    assert not app.exception and app.metric[0].value == "1"
    assert app.metric[1].value == "N/A"
