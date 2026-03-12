"""
Phase 4 — Airflow DAG
US Accidents Pipeline Orchestration
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="us_accidents_pipeline",
    description="US Accidents End-to-End Data Pipeline",
    schedule="@daily",
    default_args=default_args,
    catchup=False,
    tags=["data_engineering", "spark", "postgres"],
) as dag:

    # -------------------------------
    # Task 1: Check dataset exists
    # -------------------------------
    def check_data():
        import os

        path = "/opt/airflow/data/processed/accidents_clean.parquet"

        if os.path.exists(path):
            logger.info(f"Dataset found at {path}")
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    check_data_task = PythonOperator(
        task_id="check_data_exists",
        python_callable=check_data,
    )

    # -------------------------------
    # Task 2: Validate data quality
    # -------------------------------
    def validate_data():
        import pandas as pd

        df = pd.read_parquet("/opt/airflow/data/processed/accidents_clean.parquet")

        if len(df) == 0:
            raise ValueError("Dataset is empty")

        if "Severity" not in df.columns:
            raise ValueError("Severity column missing")

        # Remove rows where Severity is null
        df = df.dropna(subset=["Severity"])

        null_pct = df.isnull().mean().max()

        if null_pct > 0.30:
            logger.warning(f"High null percentage detected: {null_pct:.2%}")

        logger.info(f"Records after cleaning: {len(df)}")

    validate_task = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data,
    )

    # -------------------------------
    # Task 3: Generate statistics
    # -------------------------------
    def generate_stats():
        import pandas as pd
        import json
        import os

        df = pd.read_parquet("/opt/airflow/data/processed/accidents_clean.parquet")

        stats = {
            "total_records": len(df),
            "severity_distribution": df["Severity"].value_counts().to_dict(),
            "top_5_states": df["State"].value_counts().head(5).to_dict(),
        }

        os.makedirs("/opt/airflow/data/warehouse", exist_ok=True)

        with open("/opt/airflow/data/warehouse/pipeline_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("Statistics generated")

    stats_task = PythonOperator(
        task_id="generate_summary_stats",
        python_callable=generate_stats,
    )

    # -------------------------------
    # Task 4: Export dataset for BI
    # -------------------------------
    def export_powerbi():
        import pandas as pd
        import os

        df = pd.read_parquet("/opt/airflow/data/processed/accidents_clean.parquet")

        out = "/opt/airflow/data/warehouse"
        os.makedirs(out, exist_ok=True)

        df.sample(min(50000, len(df))).to_csv(
            f"{out}/accidents_dashboard.csv",
            index=False
        )

        logger.info("Power BI export created")

    export_task = PythonOperator(
        task_id="export_for_powerbi",
        python_callable=export_powerbi,
    )

    # -------------------------------
    # Task 5: Pipeline completion
    # -------------------------------
    def pipeline_complete():
        logger.info("US ACCIDENTS PIPELINE COMPLETE")

    complete_task = PythonOperator(
        task_id="pipeline_complete",
        python_callable=pipeline_complete,
    )

    # DAG order
    check_data_task >> validate_task >> stats_task >> export_task >> complete_task