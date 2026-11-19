"""Build opening-time features and a resolution label for historical tickets."""

import pandas as pd


OPENING_CATEGORIES = ("category", "product", "channel")
OPENING_COLUMNS = ("created_at", *OPENING_CATEGORIES)
FAST_THRESHOLD_HOURS = 24
TARGET = "is_fast_resolution"
RESOLUTION_FLAGS = (
    "flag_first_resolution_before_creation",
    "flag_final_resolution_before_creation",
    "flag_final_resolution_before_first",
    "flag_wallclock_timestamp_mismatch",
)


def build_opening_features(tickets: pd.DataFrame) -> pd.DataFrame:
    """Convert the same opening-time fields for training and new tickets."""
    missing = sorted(set(OPENING_COLUMNS) - set(tickets.columns))
    if missing:
        raise ValueError(f"Missing opening-time columns: {', '.join(missing)}")

    created = pd.to_datetime(tickets["created_at"], format="mixed", errors="raise")
    features = tickets.loc[:, list(OPENING_CATEGORIES)].astype("string")
    for column in OPENING_CATEGORIES:
        values = features[column].str.strip()
        features[column] = values.mask(values.eq(""))

    features["hour_created"] = created.dt.hour.astype("Int8")
    features["day_of_week"] = created.dt.day_of_week.astype("Int8")
    features["month_created"] = created.dt.month.astype("Int8")
    features["is_weekend"] = features["day_of_week"].ge(5)
    return features


def build_resolution_target(tickets: pd.DataFrame) -> pd.Series:
    """Label eligible observed outcomes and leave the rest missing."""
    hours = tickets["resolution_wallclock_hours"]
    flags = tickets.loc[:, list(RESOLUTION_FLAGS)]
    eligible = (
        hours.notna()
        & hours.ge(0)
        & tickets["status"].isin(["Closed", "Resolved"])
        & tickets["created_at"].notna()
        & tickets["resolved_at"].notna()
        & flags.notna().all(axis=1)
        & ~flags.any(axis=1)
    )
    return hours.lt(FAST_THRESHOLD_HOURS).astype("Int8").where(eligible).rename(TARGET)
