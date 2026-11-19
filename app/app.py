"""Streamlit dashboard for historical IT support analytics."""

import os

import altair as alt
import pandas as pd
import psycopg
import streamlit as st


DURATION = "resolution_wallclock_hours"
LABEL = "is_fast_resolution"
FILTERS = {
    "category": "Category",
    "priority": "Priority",
    "assigned_team": "Assigned team",
    "channel": "Channel",
}


def read_tickets(password):
    settings = {"password": password} if password else {}
    with psycopg.connect(
        dbname="it_support_analytics",
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        connect_timeout=5,
        **settings,
    ) as connection:
        cursor = connection.execute("""
            SELECT created_at, category, priority, assigned_team, channel,
                   resolution_wallclock_hours, is_fast_resolution
            FROM tickets
        """)
        return pd.DataFrame(cursor.fetchall(), columns=[column.name for column in cursor.description])


def metrics(tickets):
    mean = tickets[DURATION].mean()
    labels = tickets[LABEL].dropna()
    total, average, fast, slow = st.columns(4)
    total.metric("Total tickets", f"{len(tickets):,}")
    average.metric("Average resolution time", f"{mean:,.1f} h" if pd.notna(mean) else "N/A")
    fast.metric("Fast resolution rate", f"{labels.mean():.1%}" if len(labels) else "N/A")
    slow.metric("Slow resolution rate", f"{1 - labels.mean():.1%}" if len(labels) else "N/A")
    st.caption(
        f"{tickets[DURATION].count():,} recorded durations · "
        f"{len(labels):,} eligible labels · Fast means under 24 hours."
    )


def horizontal_chart(series, label, title, units, color="#315b79"):
    frame = series.dropna().rename("value").rename_axis("group").reset_index()
    if frame.empty:
        st.caption(f"{title}: no observations in this selection.")
        return
    chart = alt.Chart(frame, title=title).mark_bar(color=color).encode(
        x=alt.X("value:Q", title=units),
        y=alt.Y("group:N", title=label, sort="-x"),
        tooltip=[alt.Tooltip("group:N", title=label), alt.Tooltip("value:Q", title=units, format=",.1f")],
    ).properties(height=max(160, 23 * len(frame)))
    st.altair_chart(chart, width="stretch")


def overview(tickets):
    metrics(tickets)
    left, right = st.columns(2)
    with left:
        horizontal_chart(tickets["category"].value_counts(), "Category", "Tickets by category", "Tickets")
    with right:
        horizontal_chart(tickets["priority"].value_counts(), "Priority", "Tickets by priority", "Tickets")
    left, right = st.columns(2)
    with left:
        horizontal_chart(
            tickets.groupby("priority")[DURATION].mean(), "Priority",
            "Average recorded resolution by priority", "Hours", "#95683d",
        )
    with right:
        months = (tickets.groupby(tickets["created_at"].dt.to_period("M"))
                  .size().rename("tickets").rename_axis("month").reset_index())
        months["month"] = months["month"].dt.to_timestamp()
        chart = alt.Chart(months, title="Monthly ticket intake").mark_line(
            point=True, color="#315b79"
        ).encode(
            x=alt.X("month:T", title="Creation month"),
            y=alt.Y("tickets:Q", title="Tickets"),
            tooltip=[alt.Tooltip("month:T", format="%B %Y"), "tickets:Q"],
        ).properties(height=230)
        st.altair_chart(chart, width="stretch")


def ticket_analytics(tickets):
    filtered = tickets
    slots = st.columns(2)
    for index, (column, label) in enumerate(FILTERS.items()):
        options = sorted(tickets[column].dropna().unique())
        selected = slots[index % 2].multiselect(label, options)
        if selected:
            filtered = filtered.loc[filtered[column].isin(selected)]

    created = tickets["created_at"]
    bounds = (created.min().date(), created.max().date())
    dates = st.date_input("Creation date range", value=bounds, min_value=bounds[0], max_value=bounds[1])
    if len(dates) != 2:
        st.info("Select both a start and an end date.")
        return
    start, end = map(pd.Timestamp, dates)
    filtered = filtered.loc[
        filtered["created_at"].ge(start)
        & filtered["created_at"].lt(end + pd.Timedelta(days=1))
    ]
    metrics(filtered)
    if filtered.empty:
        st.info("No tickets match the selected filters.")
        return

    breakdown = filtered.groupby("category", dropna=False).agg(
        tickets=("created_at", "size"),
        recorded_durations=(DURATION, "count"),
        median_hours=(DURATION, "median"),
    ).sort_values("tickets", ascending=False)
    horizontal_chart(breakdown["tickets"], "Category", "Filtered ticket demand", "Tickets")
    st.dataframe(
        breakdown.rename(columns={
            "tickets": "Tickets",
            "recorded_durations": "Recorded durations",
            "median_hours": "Median hours",
        }).round(2),
        width="stretch",
    )


def main():
    st.set_page_config(page_title="IT Support Analytics", layout="wide")
    st.title("IT Support Analytics")
    page = st.sidebar.radio(
        "Page", ["Overview", "Ticket Analytics", "Resolution Prediction", "Model Insights"]
    )
    if page in {"Resolution Prediction", "Model Insights"}:
        from model_pages import model_insights, resolution_prediction

        if page == "Resolution Prediction":
            resolution_prediction()
        else:
            model_insights()
        return
    if "tickets" in st.session_state and st.sidebar.button("Disconnect"):
        del st.session_state["tickets"]

    if "tickets" not in st.session_state:
        with st.form("database_connection", clear_on_submit=True):
            password = st.text_input("PostgreSQL password", type="password")
            submitted = st.form_submit_button("Connect")
        if not submitted:
            return
        try:
            st.session_state["tickets"] = read_tickets(password)
        except psycopg.Error:
            st.error("Could not load tickets. Check your PostgreSQL password and database setup.")
            return

    tickets = st.session_state["tickets"]
    if tickets.empty:
        st.info("The tickets table is empty. Run the PostgreSQL loader and reconnect.")
        return
    if page == "Overview":
        overview(tickets)
    else:
        ticket_analytics(tickets)


if __name__ == "__main__":
    main()
