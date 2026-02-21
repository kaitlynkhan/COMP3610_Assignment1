import streamlit as st
import polars as pl
import plotly.express as px
import duckdb
import pandas as pd
from datetime import date

st.set_page_config(page_title="NYC Taxi Dashboard", layout="wide")


#load data
@st.cache_data
def load_data():
    trip_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    zone_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

    trips = pl.read_parquet(trip_url)
    zones = pl.read_csv(zone_url)

    trips = trips.drop_nulls([
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "trip_distance"
    ])

    trips = trips.filter(
        (pl.col("trip_distance") > 0) &
        (pl.col("fare_amount") >= 0) &
        (pl.col("fare_amount") <= 500)
    )

    trips = trips.filter(
        pl.col("tpep_dropoff_datetime") >= pl.col("tpep_pickup_datetime")
    )

    trips = trips.with_columns([
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
         .dt.total_seconds() / 60).alias("trip_duration_minutes"),
        (pl.col("trip_distance") /
         ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
          .dt.total_seconds() / 3600)).alias("trip_speed_mph"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_day_of_week")
    ])

    return trips, zones

trips_pl, zones_pl = load_data()

# Convert to pandas for Plotly + DuckDB
trips = trips_pl.to_pandas()
zones = zones_pl.to_pandas()

# Ensure datetime type
trips["tpep_pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"])
trips["tpep_dropoff_datetime"] = pd.to_datetime(trips["tpep_dropoff_datetime"])



# Title + Intro
st.title("NYC Yellow Taxi Trips (Jan 2024) Dashboard")
st.write(
    "This dashboard explores NYC Yellow Taxi trip patterns for January 2024. "
    "You can filter by date range, pickup hour, and payment type to see how trip volume and fares change."
)

# Sidebar Filters
st.sidebar.header("Filters")

min_date = trips["tpep_pickup_datetime"].dt.date.min()
max_date = trips["tpep_pickup_datetime"].dt.date.max()

date_range = st.sidebar.date_input(
    "Pickup date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

hour_range = st.sidebar.slider("Pickup hour range", 0, 23, (0, 23))

payment_options = sorted(trips["payment_type"].dropna().unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Payment types",
    options=payment_options,
    default=payment_options
)

# Apply filters
start_date, end_date = date_range
mask = (
    (trips["tpep_pickup_datetime"].dt.date >= start_date) &
    (trips["tpep_pickup_datetime"].dt.date <= end_date) &
    (trips["pickup_hour"] >= hour_range[0]) &
    (trips["pickup_hour"] <= hour_range[1]) &
    (trips["payment_type"].isin(selected_payments))
)
df = trips.loc[mask].copy()


# Key Metrics
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total_trips = len(df)
avg_fare = df["fare_amount"].mean() if total_trips > 0 else 0
total_revenue = df["total_amount"].sum() if total_trips > 0 else 0
avg_distance = df["trip_distance"].mean() if total_trips > 0 else 0
avg_duration = df["trip_duration_minutes"].mean() if total_trips > 0 else 0

col1.metric("Total Trips", f"{total_trips:,}")
col2.metric("Average Fare ($)", f"{avg_fare:.2f}")
col3.metric("Total Revenue ($)", f"{total_revenue:,.2f}")
col4.metric("Avg Trip Distance (mi)", f"{avg_distance:.2f}")
col5.metric("Avg Trip Duration (min)", f"{avg_duration:.2f}")

st.divider()


# DuckDB connection for zone joins
con = duckdb.connect()
con.register("trips", df)
con.register("zones", zones)


#5 Visualizations

# 1) Top 10 pickup zones (bar)
st.subheader("1) Top 10 Pickup Zones")
q1 = """
SELECT z.Zone AS pickup_zone, COUNT(*) AS trip_count
FROM trips t
JOIN zones z ON t.PULocationID = z.LocationID
GROUP BY pickup_zone
ORDER BY trip_count DESC
LIMIT 10;
"""
top_zones = con.execute(q1).df()

fig1 = px.bar(top_zones, x="pickup_zone", y="trip_count")
st.plotly_chart(fig1, use_container_width=True)
st.write(
    "Midtown and airport zones like JFK consistently appear among the busiest pickup areas. "
    "This suggests that business districts and transportation hubs drive a large share of total trip volume in NYC."
)

# 2) Average fare by hour (line)
st.subheader("2) Average Fare by Pickup Hour")
q2 = """
SELECT pickup_hour, AVG(fare_amount) AS avg_fare
FROM trips
GROUP BY pickup_hour
ORDER BY pickup_hour;
"""
fare_by_hour = con.execute(q2).df()

fig2 = px.line(fare_by_hour, x="pickup_hour", y="avg_fare", markers=True)
st.plotly_chart(fig2, use_container_width=True)
st.write(
    "Average fares peak during early morning and late afternoon hours. "
    "This likely reflects commuter demand and airport traffic, where longer trips increase the average fare."
)


# 3) Trip distance distribution (histogram)
st.subheader("3) Trip Distance Distribution")
#limited the extreme outliers for vizualization 
distance_filtered = df[df["trip_distance"] <= 30]
fig3 = px.histogram(distance_filtered, x="trip_distance", nbins=60)
st.plotly_chart(fig3, use_container_width=True)

st.write(
    "The majority of trips are short-distance rides under a few miles. "
    "The long right tail indicates fewer but higher-distance trips, likely airport or cross-borough rides."
)

# 4) Payment type breakdown (pie)
st.subheader("4) Payment Type Breakdown")
pay_counts = df["payment_type"].value_counts().reset_index()
pay_counts.columns = ["payment_type", "trip_count"]

fig4 = px.pie(pay_counts, names="payment_type", values="trip_count")
st.plotly_chart(fig4, use_container_width=True)
st.write(
    "Credit card payments dominate taxi transactions, accounting for the vast majority of trips. "
    "Cash usage is significantly lower, indicating strong adoption of digital payment methods in NYC."
)


# 5) Trips by day of week and hour (heatmap)
st.subheader("5) Trips by Day of Week and Hour")
q5 = """
SELECT pickup_day_of_week, pickup_hour, COUNT(*) AS trip_count
FROM trips
GROUP BY pickup_day_of_week, pickup_hour
ORDER BY pickup_day_of_week, pickup_hour;
"""
heat = con.execute(q5).df()

pivot = heat.pivot(index="pickup_day_of_week", columns="pickup_hour", values="trip_count").fillna(0)

fig5 = px.imshow(pivot, aspect="auto")
st.plotly_chart(fig5, use_container_width=True)
st.write(
    "Trip demand increases during weekday daytime hours and peaks in late afternoon periods. "
    "Weekend patterns show higher late-night activity, reflecting leisure travel behavior."
)



