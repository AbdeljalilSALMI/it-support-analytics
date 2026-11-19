"""Clean ticket data and export a typed, analysis-ready Parquet file."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


DATE_COLUMNS = ("created_at", "first_resolved_at", "resolved_at", "closed_at")
INTEGER_COLUMNS = (
    "customer_tenure_months", "previous_tickets", "issue_complexity_score", "reopen_count",
)
NUMBER_COLUMNS = (
    "first_response_time_hours", "resolution_time_hours", "paused_hours",
    "resolution_wallclock_hours", "sla_target_hours", "sla_breach_margin",
    "customer_satisfaction_score",
)
CATEGORY_COLUMNS = (
    "category", "issue_type_id", "product", "channel", "region", "language",
    "subscription_type", "customer_segment", "priority", "assigned_team", "status",
    "escalation_reason",
)
BOOLEAN_COLUMNS = ("escalated", "sla_breached")
OUTPUT_COLUMNS = (
    "ticket_id", *DATE_COLUMNS, *CATEGORY_COLUMNS, *INTEGER_COLUMNS, *NUMBER_COLUMNS,
    *BOOLEAN_COLUMNS,
)
DURATION_TOLERANCE_HOURS = 0.01


def reject_invalid(source: pd.Series, parsed: pd.Series) -> None:
    invalid_count = (source.notna() & parsed.isna()).sum()
    if invalid_count:
        raise ValueError(f"{source.name}: {invalid_count:,} nonblank values could not be parsed.")


def clean_tickets(raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve ticket records and flag suspect observations for later review."""
    missing_columns = sorted(set(OUTPUT_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Check exact source duplicates before projecting away personal and text fields.
    tickets = raw.drop_duplicates().loc[:, list(OUTPUT_COLUMNS)].copy()
    for column in OUTPUT_COLUMNS:
        values = tickets[column].astype("string").str.strip()
        tickets[column] = values.mask(values.eq(""))

    ids = tickets["ticket_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Ticket IDs must be present and unique after removing exact duplicates.")

    for column in DATE_COLUMNS:
        source = tickets[column]
        parsed = pd.to_datetime(source, format="mixed", errors="coerce")
        reject_invalid(source, parsed)
        tickets[column] = parsed.dt.as_unit("ns")

    for column in (*INTEGER_COLUMNS, *NUMBER_COLUMNS):
        source = tickets[column]
        parsed = pd.to_numeric(source, errors="coerce")
        parsed = parsed.where(np.isfinite(parsed))
        reject_invalid(source, parsed)
        if column in INTEGER_COLUMNS and parsed.dropna().mod(1).ne(0).any():
            raise ValueError(f"{column}: expected whole numbers.")
        tickets[column] = parsed.astype("Int64" if column in INTEGER_COLUMNS else "Float64")

    for column in CATEGORY_COLUMNS:
        source = tickets[column]
        # Keep an entirely missing column as nullable text for a stable Parquet round trip.
        if source.notna().any():
            tickets[column] = source.astype("str").astype("category")

    for column in BOOLEAN_COLUMNS:
        source = tickets[column]
        parsed = source.str.casefold().map({"yes": True, "no": False}).astype("boolean")
        reject_invalid(source, parsed)
        tickets[column] = parsed

    elapsed_hours = (tickets["resolved_at"] - tickets["created_at"]).dt.total_seconds() / 3600
    duration_gap = tickets["resolution_wallclock_hours"].sub(elapsed_hours).abs()

    tickets["has_recorded_wallclock_duration"] = tickets["resolution_wallclock_hours"].notna()
    tickets["flag_negative_resolution_time"] = tickets["resolution_time_hours"].lt(0).fillna(False)
    tickets["flag_first_resolution_before_creation"] = tickets["first_resolved_at"].lt(tickets["created_at"])
    tickets["flag_final_resolution_before_creation"] = elapsed_hours.lt(0)
    tickets["flag_final_resolution_before_first"] = tickets["resolved_at"].lt(tickets["first_resolved_at"])
    tickets["flag_closure_before_final_resolution"] = tickets["closed_at"].lt(tickets["resolved_at"])
    tickets["flag_wallclock_timestamp_mismatch"] = duration_gap.gt(DURATION_TOLERANCE_HOURS).fillna(False)
    return tickets.reset_index(drop=True)


def read_raw_tickets(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def write_processed_tickets(tickets: pd.DataFrame, path: Path) -> None:
    """Replace the output only after the new Parquet file is fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        tickets.to_parquet(temporary_path, engine="pyarrow", index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project.parent / "tickets.csv")
    parser.add_argument("--output", type=Path, default=project / "data/processed/tickets.parquet")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"CSV file not found: {args.input}")
    if args.output.suffix.lower() != ".parquet" or args.output.resolve() == args.input.resolve():
        parser.error("Choose a separate output path ending in .parquet.")

    raw = read_raw_tickets(args.input)
    tickets = clean_tickets(raw)
    write_processed_tickets(tickets, args.output)
    print(f"Input: {len(raw):,} rows. Output: {len(tickets):,} rows, {len(tickets.columns)} columns.")
    print(f"Exact duplicates removed: {len(raw) - len(tickets):,}")
    print(f"Missing wall-clock durations: {(~tickets['has_recorded_wallclock_duration']).sum():,}")
    print(tickets.filter(regex="^flag_").sum().to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
