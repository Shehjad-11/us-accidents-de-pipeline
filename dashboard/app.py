"""
US Accidents Analytics Dashboard
Big Data Pipeline: Kafka → Spark → PostgreSQL → XGBoost → Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="US Accidents Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 US Traffic Accidents Dashboard")
st.markdown(
    "Interactive analytics dashboard powered by **XGBoost Machine Learning Model**"
)


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/accidents_clean.parquet")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df = df[df["State"].notna()]
    return df


@st.cache_resource
def load_model():
    path = "ml/models/xgboost_severity.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


df = load_data()
model = load_model()


# ---------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------

st.sidebar.header("Filters")

state_options = sorted(df["State"].unique())

default_states = (
    df["State"]
    .value_counts()
    .head(3)
    .index
    .tolist()
)

selected_states = st.sidebar.multiselect(
    "Select States",
    options=state_options,
    default=default_states
)

selected_severity = st.sidebar.multiselect(
    "Select Severity Levels",
    options=[1, 2, 3, 4],
    default=[1, 2, 3, 4]
)

filtered_df = df[
    df["State"].isin(selected_states) &
    df["Severity"].isin(selected_severity)
]


# ---------------------------------------------------
# KPI Metrics
# ---------------------------------------------------

st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accidents", f"{len(filtered_df):,}")
col2.metric("Average Severity", f"{filtered_df['Severity'].mean():.2f}")
col3.metric("Rush Hour %", f"{filtered_df['is_rush_hour'].mean()*100:.1f}%")
col4.metric("Night Accidents %", f"{filtered_df['is_night'].mean()*100:.1f}%")

st.divider()


# ---------------------------------------------------
# Severity Distribution
# ---------------------------------------------------

st.subheader("Accident Severity Distribution")

severity_counts = (
    filtered_df["Severity"]
    .value_counts()
    .sort_index()
    .reset_index()
)

severity_counts.columns = ["Severity", "Count"]

severity_counts["Label"] = severity_counts["Severity"].map({
    1: "Minor",
    2: "Moderate",
    3: "Serious",
    4: "Critical"
})

fig = px.bar(
    severity_counts,
    x="Label",
    y="Count",
    color="Count",
    color_continuous_scale="RdYlGn_r"
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# Hourly Accident Trends
# ---------------------------------------------------

st.subheader("Accidents by Hour")

hour_data = (
    filtered_df
    .groupby("hour_of_day")
    .size()
    .reset_index(name="count")
)

fig = px.line(
    hour_data,
    x="hour_of_day",
    y="count",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# Weekly Pattern
# ---------------------------------------------------

st.subheader("Accidents by Day of Week")

day_map = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun"
}

day_data = (
    filtered_df
    .groupby("day_of_week")
    .size()
    .reset_index(name="count")
)

day_data["day"] = day_data["day_of_week"].map(day_map)

fig = px.bar(
    day_data,
    x="day",
    y="count",
    color="count"
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# Accident Map
# ---------------------------------------------------

st.subheader("Accident Locations")

map_data = filtered_df[
    ["Start_Lat", "Start_Lng", "Severity"]
].dropna()

if len(map_data) > 0:

    sample_map = map_data.sample(
        min(5000, len(map_data))
    )

    fig = px.scatter_mapbox(
        sample_map,
        lat="Start_Lat",
        lon="Start_Lng",
        color="Severity",
        zoom=3,
        height=500,
        mapbox_style="carto-positron"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# Machine Learning Prediction Tool
# ---------------------------------------------------

st.divider()
st.subheader("🤖 Accident Severity Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider("Hour of Day", 0, 23, 8)
    day = st.slider("Day of Week", 0, 6, 1)
    month = st.slider("Month", 1, 12, 6)

with col2:
    temperature = st.slider("Temperature (°F)", 0, 120, 65)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    visibility = st.slider("Visibility (mi)", 0, 10, 8)

with col3:
    wind_speed = st.slider("Wind Speed (mph)", 0, 100, 10)
    duration = st.slider("Accident Duration (min)", 0, 120, 30)


if st.button("Predict Severity"):

    if model:

        features = pd.DataFrame([{
            "hour_of_day": hour,
            "day_of_week": day,
            "month": month,
            "year": 2024,
            "is_weekend": 0,
            "is_rush_hour": 0,
            "is_night": 0,
            "duration_mins": duration,
            "Start_Lat": 37.5,
            "Start_Lng": -120.0,
            "Temperature(F)": temperature,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": 0,
            "Amenity": 0,
            "Crossing": 0,
            "Junction": 0,
            "Railway": 0,
            "Station": 0,
            "Stop": 0,
            "Traffic_Signal": 0
        }])

        prediction = model.predict(features)[0] + 1

        severity_map = {
            1: "🟢 Minor",
            2: "🟡 Moderate",
            3: "🔴 Serious",
            4: "⚫ Critical"
        }

        st.success(f"Predicted Severity: {severity_map[prediction]}")

    else:
        st.error("Model not found. Train the model first.")


# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()
st.caption(
    "Pipeline: Kaggle Dataset → Kafka → Spark → PostgreSQL → XGBoost → Streamlit"
)