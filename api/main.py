"""
Phase 7 — FastAPI
US Accidents REST API
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

# ── App Setup ──────────────────────────────────────────
app = FastAPI(
    title="US Accidents API",
    description="REST API for US Accidents Data Pipeline & Severity Prediction",
    version="1.0.0"
)

# ── Load Data & Model ──────────────────────────────────
@app.on_event("startup")
async def startup():
    global df, model

    # Load dataset
    df = pd.read_parquet("data/processed/accidents_clean.parquet")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])

    # Load model
    model_path = "ml/models/xgboost_severity.pkl"
    model = joblib.load(model_path) if os.path.exists(model_path) else None

    print(f"✅ Loaded {len(df):,} records")
    print(f"✅ Model loaded: {model is not None}")

# ── Request Schema ─────────────────────────────────────
class PredictionRequest(BaseModel):
    hour_of_day:      int   = 8
    day_of_week:      int   = 2
    month:            int   = 6
    is_weekend:       int   = 0
    is_rush_hour:     int   = 1
    is_night:         int   = 0
    duration_mins:    float = 30.0
    temperature_f:    float = 65.0
    humidity_pct:     float = 50.0
    visibility_mi:    float = 10.0
    wind_speed_mph:   float = 10.0
    precipitation_in: float = 0.0
    start_lat:        float = 37.5
    start_lng:        float = -120.0
    junction:         int   = 0
    traffic_signal:   int   = 0
    crossing:         int   = 0

# ── Endpoints ──────────────────────────────────────────

# 1. Health Check
@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "records":      len(df),
        "model_loaded": model is not None,
        "version":      "1.0.0"
    }

# 2. Pipeline Stats
@app.get("/stats")
def get_stats():

    return {
        "total_records": int(len(df)),

        "severity_distribution":
            df["Severity"].value_counts().sort_index().astype(int).to_dict(),

        "top_5_states":
            df["State"].value_counts().head(5).astype(int).to_dict(),

        "rush_hour_pct":
            float(round(df["is_rush_hour"].mean() * 100, 2)),

        "night_accident_pct":
            float(round(df["is_night"].mean() * 100, 2)),

        "avg_duration_mins":
            float(round(df["duration_mins"].mean(), 2)),

        "avg_temperature_f":
            float(round(df["Temperature(F)"].mean(), 2))
    }

# 3. Get Accidents (with optional state filter)
@app.get("/accidents")
def get_accidents(
    state:    str = Query(None, description="Filter by state e.g. CA"),
    severity: int = Query(None, description="Filter by severity 1-4"),
    limit:    int = Query(100,  description="Max records to return")
):
    result = df.copy()

    if state:
        state = state.upper()
        if state not in df["State"].values:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found")
        result = result[result["State"] == state]

    if severity:
        if severity not in [1, 2, 3, 4]:
            raise HTTPException(status_code=400, detail="Severity must be 1, 2, 3 or 4")
        result = result[result["Severity"] == severity]

    result = result.head(limit)

    return {
        "total":   len(result),
        "state":   state,
        "data": result[[
            "ID", "Severity", "State", "City",
            "Start_Lat", "Start_Lng",
            "Weather_Condition", "hour_of_day",
            "is_rush_hour", "duration_mins"
        ]].fillna("").to_dict(orient="records")
    }

# 4. Severity Stats by State
@app.get("/accidents/by-state")
def accidents_by_state():
    result = df.groupby("State").agg(
        total=("Severity", "count"),
        avg_severity=("Severity", "mean"),
        high_severity=("severity_binary", "sum")
    ).reset_index()
    result["avg_severity"] = result["avg_severity"].round(2)
    return result.sort_values("total", ascending=False).to_dict(orient="records")

# 5. Time Pattern Analysis
@app.get("/accidents/by-hour")
def accidents_by_hour():
    result = df.groupby("hour_of_day").agg(
        count=("Severity", "count"),
        avg_severity=("Severity", "mean")
    ).reset_index()
    result["avg_severity"] = result["avg_severity"].round(2)
    return result.to_dict(orient="records")

# 6. Severity Prediction
@app.post("/predict")
def predict_severity(req: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = pd.DataFrame([{
        "hour_of_day":       req.hour_of_day,
        "day_of_week":       req.day_of_week,
        "month":             req.month,
        "year":              2024,
        "is_weekend":        req.is_weekend,
        "is_rush_hour":      req.is_rush_hour,
        "is_night":          req.is_night,
        "duration_mins":     req.duration_mins,
        "Start_Lat":         req.start_lat,
        "Start_Lng":         req.start_lng,
        "Temperature(F)":    req.temperature_f,
        "Humidity(%)":       req.humidity_pct,
        "Visibility(mi)":    req.visibility_mi,
        "Wind_Speed(mph)":   req.wind_speed_mph,
        "Precipitation(in)": req.precipitation_in,
        "Amenity":           0,
        "Crossing":          req.crossing,
        "Junction":          req.junction,
        "Railway":           0,
        "Station":           0,
        "Stop":              0,
        "Traffic_Signal":    req.traffic_signal,
    }])

    pred  = model.predict(features)[0] + 1
    proba = model.predict_proba(features)[0]

    severity_map = {
        1: "Minor",
        2: "Moderate",
        3: "Serious",
        4: "Critical"
    }

    return {
        "predicted_severity":       int(pred),
        "severity_label":           severity_map[int(pred)],
        "confidence_pct":           round(float(proba[pred - 1]) * 100, 2),
        "probabilities": {
            "Minor":    round(float(proba[0]) * 100, 2),
            "Moderate": round(float(proba[1]) * 100, 2),
            "Serious":  round(float(proba[2]) * 100, 2),
            "Critical": round(float(proba[3]) * 100, 2),
        },
        "input": req.dict()
    }