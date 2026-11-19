"""Print aggregate checks for the ticket dataset used in Step 2."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_csv = Path(__file__).resolve().parents[2] / "tickets.csv"
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=default_csv,
        help="Path to the ticket CSV (defaults to tickets.csv in the parent workspace).",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}. Pass a path to an existing CSV.")

    tickets = pd.read_csv(args.csv, dtype="string", keep_default_na=False)
    print(f"{len(tickets):,} rows, {len(tickets.columns)} columns")
    print(f"Exact duplicate rows: {tickets.duplicated().sum():,}")

    # Preserve source strings and treat only blank or whitespace-only cells as missing.
    tickets = tickets.replace(r"^\s*$", pd.NA, regex=True)
    print(f"Duplicate nonblank IDs: {tickets['ticket_id'].dropna().duplicated().sum():,}")
    missing = tickets.isna().sum()
    print("\nColumns with missing values")
    print(missing[missing.gt(0)].sort_values(ascending=False).to_string())

    dates = {}
    for column in ("created_at", "resolved_at", "closed_at"):
        dates[column] = pd.to_datetime(tickets[column], format="mixed", errors="coerce")
        invalid = dates[column].isna() & tickets[column].notna()
        print(f"{column}: {dates[column].min()} to {dates[column].max()}, "
              f"{invalid.sum():,} unparseable nonblank values")

    print("\nRecorded resolution duration by ticket status")
    print(pd.crosstab(tickets["status"], tickets["resolution_time_hours"].notna())
          .rename(columns={False: "Missing", True: "Present"}).to_string())

    for column in ("resolution_time_hours", "resolution_wallclock_hours"):
        hours = pd.to_numeric(tickets[column], errors="coerce")
        invalid = hours.isna() & tickets[column].notna()
        print(f"{column}: {invalid.sum():,} unparseable, {hours.lt(0).sum():,} negative, "
              f"{hours.between(0, 24, inclusive='left').sum():,} in [0, 24), "
              f"{hours.ge(24).sum():,} at least 24")

    print(f"Resolved before creation: {dates['resolved_at'].lt(dates['created_at']).sum():,}")
    print(f"Closed before resolution: {dates['closed_at'].lt(dates['resolved_at']).sum():,}")


if __name__ == "__main__":
    main()
