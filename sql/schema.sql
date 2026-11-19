CREATE TABLE IF NOT EXISTS tickets (
    ticket_id text PRIMARY KEY,
    created_at timestamp NOT NULL,
    category text,
    issue_type_id text,
    product text,
    channel text,
    region text,
    priority text,
    assigned_team text,
    status text NOT NULL,
    resolved_at timestamp,
    resolution_wallclock_hours double precision CHECK (resolution_wallclock_hours >= 0),
    is_fast_resolution boolean,
    CHECK (is_fast_resolution IS NULL OR resolution_wallclock_hours IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS tickets_created_at_idx ON tickets (created_at);
