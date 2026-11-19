from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import pytest

from src.features import build_opening_features, build_resolution_target
from src.prediction import explain_slow_resolution
from src.preprocessing import OUTPUT_COLUMNS, clean_tickets, write_processed_tickets


PROJECT = Path(__file__).resolve().parents[1]


def example_ticket():
    row = dict.fromkeys(OUTPUT_COLUMNS, "")
    row.update(
        ticket_id="001", created_at="2024-06-08 14:30", status="Closed",
        category=" Billing ", product="API Platform", channel="Email",
        resolved_at="2024-06-09 14:30", resolution_wallclock_hours="24",
    )
    return pd.DataFrame([row])


def test_cleaning_deduplicates_and_preserves_missing_values():
    raw = example_ticket()
    cleaned = clean_tickets(pd.concat([raw, raw], ignore_index=True))
    assert cleaned["ticket_id"].tolist() == ["001"]
    assert cleaned["category"].tolist() == ["Billing"]
    assert pd.isna(cleaned.loc[0, "escalated"])
    target = build_resolution_target(cleaned)
    assert target.tolist() == [0]
    with TemporaryDirectory(prefix=".pytest-tmp-", dir=PROJECT) as temporary:
        path = Path(temporary) / "tickets.parquet"
        write_processed_tickets(cleaned, path)
        pd.testing.assert_frame_equal(pd.read_parquet(path), cleaned)


@pytest.mark.parametrize(("column", "value"), [
    ("created_at", "invalid"),
    ("reopen_count", "1.5"),
    ("escalated", "unknown"),
    ("resolution_time_hours", "inf"),
])
def test_cleaning_rejects_malformed_values(column, value):
    with pytest.raises(ValueError, match=column):
        clean_tickets(example_ticket().assign(**{column: value}))


def test_cleaning_rejects_conflicting_ids():
    raw = example_ticket()
    with pytest.raises(ValueError, match="unique"):
        clean_tickets(pd.concat([raw, raw.assign(priority="High")], ignore_index=True))


def test_missing_outcome_remains_unlabeled():
    cleaned = clean_tickets(example_ticket().assign(
        status="Open", resolved_at="", resolution_wallclock_hours="",
    ))
    assert pd.isna(build_resolution_target(cleaned).iloc[0])


def test_saved_model_handles_missing_and_unseen_inputs():
    path = PROJECT / "models" / "resolution_classifier.joblib"
    if not path.is_file():
        pytest.skip("Generate the classifier with the machine learning notebook.")
    model = joblib.load(path)
    features = build_opening_features(example_ticket().assign(
        category="unseen category", product=pd.NA, created_at=pd.NA,
    ))
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (1, 2)
    assert probabilities.sum() == pytest.approx(1)
    baseline, contributions = explain_slow_resolution(model, features)
    assert (baseline + contributions.sum(axis=1)).iloc[0] == pytest.approx(probabilities[0, 0])
