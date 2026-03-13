"""
Migrate data to Railway PostgreSQL
"""

import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Read DATABASE_URL from environment
RAILWAY_URL = os.getenv("DATABASE_URL")

# Path to processed parquet dataset
PARQUET_PATH = "data/processed/accidents_clean.parquet"


def migrate():
    logger.info("🚀 Migrating to Railway PostgreSQL...")

    if not RAILWAY_URL:
        raise RuntimeError("DATABASE_URL not found in environment variables")

    # Create DB engine
    engine = create_engine(RAILWAY_URL)

    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("✅ Connected to Railway DB!")

    # Load dataset
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"✅ Loaded {len(df):,} records")

    # Column rename mapping
    cols = {
        "ID": "id", "Severity": "severity",
        "Start_Time": "start_time", "End_Time": "end_time",
        "Start_Lat": "start_lat", "Start_Lng": "start_lng",
        "City": "city", "State": "state",
        "Distance(mi)": "distance_mi",
        "Temperature(F)": "temperature_f",
        "Humidity(%)": "humidity_pct",
        "Visibility(mi)": "visibility_mi",
        "Wind_Speed(mph)": "wind_speed_mph",
        "Precipitation(in)": "precipitation_in",
        "Weather_Condition": "weather_condition",
        "hour_of_day": "hour_of_day",
        "day_of_week": "day_of_week",
        "month": "month", "year": "year",
        "is_weekend": "is_weekend",
        "is_rush_hour": "is_rush_hour",
        "is_night": "is_night",
        "duration_mins": "duration_mins",
        "severity_binary": "severity_binary"
    }

    available = {k: v for k, v in cols.items() if k in df.columns}
    df_clean = df[list(available.keys())].rename(columns=available)

    # Upload main table
    logger.info("📤 Uploading accidents_cleaned...")
    df_clean.to_sql(
        "accidents_cleaned",
        engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi"
    )
    logger.info(f"✅ {len(df_clean):,} rows uploaded!")

    # Severity stats
    logger.info("📤 Uploading severity_stats...")
    sev = df.rename(columns={"State": "state", "Severity": "severity"})

    severity_stats = sev.groupby(["state", "severity"]).agg(
        count=("severity", "count"),
        avg_duration=("duration_mins", "mean")
    ).reset_index()

    severity_stats.to_sql(
        "severity_stats",
        engine,
        if_exists="replace",
        index=False
    )

    logger.info("✅ severity_stats uploaded!")

    # Time patterns
    logger.info("📤 Uploading time_patterns...")

    time_pat = df.groupby(
        ["hour_of_day", "day_of_week", "is_rush_hour"]
    ).size().reset_index(name="accident_count")

    time_pat.to_sql(
        "time_patterns",
        engine,
        if_exists="replace",
        index=False
    )

    logger.info("✅ time_patterns uploaded!")

    # Verify tables
    logger.info("\n📋 VERIFYING RAILWAY TABLES:")

    with engine.connect() as conn:
        for table in ["accidents_cleaned", "severity_stats", "time_patterns"]:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.fetchone()[0]
            logger.info(f"{table:30s} → {count:,} rows ✅")

    logger.info("\n" + "=" * 50)
    logger.info("🏁 RAILWAY MIGRATION COMPLETE!")
    logger.info("=" * 50)


if __name__ == "__main__":
    migrate()