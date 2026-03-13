"""
Phase 7–8 — FastAPI + Railway PostgreSQL
US Accidents REST API
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine

# ── App Setup ──────────────────────────────────────────
app = FastAPI(
    title="US Accidents API",
    description="REST API for US Accidents Data Pipeline & Severity Prediction",
    version="1.0.0"
)

# Global objects
df = pd.DataFrame()
model = None
engine = None


# ── Load Data & Model ──────────────────────────────────
@app.on_event("startup")
async def startup():
    global df, model, engine

    DATABASE_URL = os.getenv("DATABASE_URL")

    if not DATABASE_URL:
        print("❌ DATABASE_URL environment variable not set")
        return

    try:
        print("🔌 Connecting to Railway PostgreSQL...")

        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True
        )

        df = pd.read_sql("SELECT * FROM accidents_cleaned", engine)

        print(f"✅ Loaded {len(df):,} records from Railway DB")

    except Exception as e:
        print("❌ Database connection failed")
        print(e)

    try:
        model_path = "ml/models/xgboost_severity.pkl"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model file not found")

    except Exception as e:
        print("❌ Model loading failed")
        print(e)


# ── Request Schema ─────────────────────────────────────
class PredictionRequest(BaseModel):
    hour_of_day: int = 8
    day_of_week: int = 2
    month: int = 6
    is_weekend: int = 0
    is_rush_hour: int = 1
    is_night: int = 0
    duration_mins: float = 30.0
    temperature_f: float = 65.0
    humidity_pct: float = 50.0
    visibility_mi: float = 10.0
    wind_speed_mph: float = 10.0
    precipitation_in: float = 0.0
    start_lat: float = 37.5
    start_lng: float = -120.0
    junction: int = 0
    traffic_signal: int = 0
    crossing: int = 0


# ── Endpoints ──────────────────────────────────────────

# 1️⃣ Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "records": int(len(df)) if not df.empty else 0,
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


# 2️⃣ Pipeline Stats
@app.get("/stats")
def get_stats():
    return {
        "total_records": int(len(df)),
        "severity_distribution":
            df["severity"].value_counts().sort_index().astype(int).to_dict(),
        "top_5_states":
            df["state"].value_counts().head(5).astype(int).to_dict(),
        "rush_hour_pct":
            float(round(df["is_rush_hour"].mean() * 100, 2)),
        "night_accident_pct":
            float(round(df["is_night"].mean() * 100, 2)),
        "avg_duration_mins":
            float(round(df["duration_mins"].mean(), 2)),
        "avg_temperature_f":
            float(round(df["temperature_f"].mean(), 2))
    }


# 3️⃣ Get Accidents
@app.get("/accidents")
def get_accidents(
    state: str = Query(None),
    severity: int = Query(None),
    limit: int = Query(100)
):

    result = df.copy()

    if state:
        state = state.upper()
        if state not in df["state"].values:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found")
        result = result[result["state"] == state]

    if severity:
        if severity not in [1, 2, 3, 4]:
            raise HTTPException(status_code=400, detail="Severity must be 1-4")
        result = result[result["severity"] == severity]

    result = result.head(limit)

    return {
        "total": len(result),
        "state": state,
        "data": result[[
            "id", "severity", "state", "city",
            "start_lat", "start_lng",
            "weather_condition",
            "hour_of_day",
            "is_rush_hour",
            "duration_mins"
        ]].fillna("").to_dict(orient="records")
    }


# 4️⃣ Accidents by State
@app.get("/accidents/by-state")
def accidents_by_state():
    result = df.groupby("state").agg(
        total=("severity", "count"),
        avg_severity=("severity", "mean"),
        high_severity=("severity_binary", "sum")
    ).reset_index()

    result["avg_severity"] = result["avg_severity"].round(2)

    return result.sort_values("total", ascending=False).to_dict(orient="records")


# 5️⃣ Accidents by Hour
@app.get("/accidents/by-hour")
def accidents_by_hour():
    result = df.groupby("hour_of_day").agg(
        count=("severity", "count"),
        avg_severity=("severity", "mean")
    ).reset_index()

    result["avg_severity"] = result["avg_severity"].round(2)

    return result.to_dict(orient="records")


# 6️⃣ Severity Prediction
@app.post("/predict")
def predict_severity(req: PredictionRequest):

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = pd.DataFrame([{
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "month": req.month,
        "year": 2024,
        "is_weekend": req.is_weekend,
        "is_rush_hour": req.is_rush_hour,
        "is_night": req.is_night,
        "duration_mins": req.duration_mins,
        "start_lat": req.start_lat,
        "start_lng": req.start_lng,
        "temperature_f": req.temperature_f,
        "humidity_pct": req.humidity_pct,
        "visibility_mi": req.visibility_mi,
        "wind_speed_mph": req.wind_speed_mph,
        "precipitation_in": req.precipitation_in,
        "junction": req.junction,
        "traffic_signal": req.traffic_signal,
        "crossing": req.crossing
    }])

    pred = model.predict(features)[0] + 1
    proba = model.predict_proba(features)[0]

    severity_map = {
        1: "Minor",
        2: "Moderate",
        3: "Serious",
        4: "Critical"
    }

    return {
        "predicted_severity": int(pred),
        "severity_label": severity_map[int(pred)],
        "confidence_pct": round(float(proba[pred - 1]) * 100, 2),
        "probabilities": {
            "Minor": round(float(proba[0]) * 100, 2),
            "Moderate": round(float(proba[1]) * 100, 2),
            "Serious": round(float(proba[2]) * 100, 2),
            "Critical": round(float(proba[3]) * 100, 2),
        },
        "input": req.dict()
    }