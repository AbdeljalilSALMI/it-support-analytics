"""Load processed ticket data into PostgreSQL using standard libpq settings."""

import argparse
from getpass import getpass
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from src.features import build_resolution_target


PROJECT = Path(__file__).resolve().parents[1]
DATABASE = "it_support_analytics"
COLUMNS = (
    "ticket_id", "created_at", "category", "issue_type_id", "product", "channel",
    "region", "priority", "assigned_team", "status", "resolved_at",
    "resolution_wallclock_hours", "is_fast_resolution",
)


def load_tickets(connection, tickets):
    """Stage and upsert a batch within the caller's database transaction."""
    rows = tickets.assign(is_fast_resolution=build_resolution_target(tickets).astype("boolean"))
    hours = rows["resolution_wallclock_hours"]
    rows["resolution_wallclock_hours"] = hours.where(hours.ge(0))
    rows = rows.loc[:, list(COLUMNS)].astype(object)
    rows = rows.where(rows.notna(), None)

    schema = (PROJECT / "sql" / "schema.sql").read_text(encoding="utf-8")
    identifiers = sql.SQL(", ").join(map(sql.Identifier, COLUMNS))
    updates = sql.SQL(", ").join(
        sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(column)) for column in COLUMNS[1:]
    )
    with connection.cursor() as cursor:
        cursor.execute(schema)
        cursor.execute("CREATE TEMP TABLE ticket_stage (LIKE tickets) ON COMMIT DROP")
        with cursor.copy(sql.SQL("COPY ticket_stage ({}) FROM STDIN").format(identifiers)) as copy:
            for row in rows.itertuples(index=False, name=None):
                copy.write_row(row)
        cursor.execute(
            sql.SQL("INSERT INTO tickets ({0}) SELECT {0} FROM ticket_stage ORDER BY ticket_id "
                    "ON CONFLICT (ticket_id) DO UPDATE SET {1}").format(identifiers, updates)
        )
        return cursor.rowcount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "data/processed/tickets.parquet")
    parser.add_argument("--prompt-password", action="store_true",
                        help="Prompt for the database password without echoing it.")
    args = parser.parse_args()
    tickets = pd.read_parquet(args.input)
    settings = {"password": getpass("PostgreSQL password: ")} if args.prompt_password else {}
    with psycopg.connect(dbname=DATABASE, **settings) as connection:
        count = load_tickets(connection, tickets)
    print(f"Loaded {count:,} tickets into {DATABASE}.")


if __name__ == "__main__":
    main()
