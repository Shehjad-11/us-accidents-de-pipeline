"""
Phase 5 — XGBoost Severity Classifier
Trains, evaluates and explains the ML model
"""

import pandas as pd
import numpy as np
import xgboost as xgb
# import shap
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────
DATA_PATH   = "data/processed/accidents_clean.parquet"
MODEL_DIR   = "ml/models"
REPORTS_DIR = "ml/reports"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Feature Columns ────────────────────────────────────
FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "year",
    "is_weekend", "is_rush_hour", "is_night",
    "duration_mins", "Start_Lat", "Start_Lng",
]

WEATHER_COLS = [
    "Temperature(F)", "Humidity(%)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"
]

ROAD_COLS = [
    "Amenity", "Crossing", "Junction",
    "Railway", "Station", "Stop", "Traffic_Signal"
]

TARGET = "Severity"

# ── Step 1: Load Data ──────────────────────────────────
def load_data():
    print("📂 Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Loaded {len(df):,} records")
    print(f"\n🔢 Severity Distribution:")
    print(df[TARGET].value_counts().sort_index())
    return df

# ── Step 2: Prepare Features ───────────────────────────
def prepare_features(df):
    print("\n⚙️  Preparing features...")

    # Parse time features if missing
    if "hour_of_day" not in df.columns:
        df["Start_Time"]   = pd.to_datetime(df["Start_Time"])
        df["End_Time"]     = pd.to_datetime(df["End_Time"])
        df["hour_of_day"]  = df["Start_Time"].dt.hour
        df["day_of_week"]  = df["Start_Time"].dt.dayofweek
        df["month"]        = df["Start_Time"].dt.month
        df["year"]         = df["Start_Time"].dt.year
        df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_rush_hour"] = (
            df["hour_of_day"].between(7, 9) |
            df["hour_of_day"].between(16, 19)
        ).astype(int)
        df["is_night"]     = (
            (df["hour_of_day"] >= 20) |
            (df["hour_of_day"] <= 6)
        ).astype(int)
        df["duration_mins"] = (
            pd.to_datetime(df["End_Time"]) -
            pd.to_datetime(df["Start_Time"])
        ).dt.total_seconds() / 60

    # Convert boolean road cols to int
    for col in ROAD_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Select available features
    all_features = FEATURE_COLS + WEATHER_COLS + ROAD_COLS
    available    = [f for f in all_features if f in df.columns]

    X = df[available].fillna(0)
    y = df[TARGET].astype(int) - 1  # 0-indexed: 0,1,2,3

    print(f"✅ Features ready: {len(available)} columns | {len(X):,} samples")
    print(f"   Features used: {available}")
    return X, y, available

# ── Step 3: Handle Class Imbalance ─────────────────────
def apply_smote(X, y):
    print("\n⚖️  Applying SMOTE...")
    print(f"   Before: {pd.Series(y).value_counts().sort_index().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"   After:  {pd.Series(y_res).value_counts().sort_index().to_dict()}")
    return X_res, y_res

# ── Step 4: Train XGBoost ──────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n🤖 Training XGBoost...")

    model = xgb.XGBClassifier(
        objective="multi:softprob",   # correct objective for multi-class
        num_class=4,                  # 4 severity classes
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=20,
        verbosity=1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print("✅ XGBoost training complete!")
    return model# ── Step 5: Train Random Forest ────────────────────────
def train_random_forest(X_train, y_train):
    print("\n🌲 Training Random Forest (baseline)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest complete!")
    return rf

# ── Step 6: Evaluate ───────────────────────────────────
def evaluate(model, X_test, y_test, name):
    print(f"\n📊 Evaluating {name}...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sev1','Sev2','Sev3','Sev4'])}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sev1","Sev2","Sev3","Sev4"],
                yticklabels=["Sev1","Sev2","Sev3","Sev4"])
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/{name.lower().replace(' ','_')}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"💾 Confusion matrix saved!")

    return {"model": name, "accuracy": round(acc, 4), "f1_score": round(f1, 4)}

# ── Step 7: Feature Importance (XGBoost) ─────────────────
def explain_features(model, X_test, features):
    print("\n🔍 Generating feature importance plot...")

    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({
            "feature": features,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10,6))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Top 15 Feature Importances (XGBoost)")
    plt.tight_layout()

    path = f"{REPORTS_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"✅ Feature importance saved → {path}")
# ── Main ───────────────────────────────────────────────
def main():
    print("🚀 Starting ML Pipeline...\n")

    df                    = load_data()
    X, y, features        = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    X_train, X_val, y_train, y_val   = train_test_split(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_STATE, stratify=y_train)

    # SMOTE on training only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Train models
    xgb_model = train_xgboost(X_train_res, y_train_res, X_val, y_val)
    rf_model  = train_random_forest(X_train_res, y_train_res)

    # Evaluate models
    results = []
    results.append(evaluate(xgb_model, X_test, y_test, "XGBoost"))
    results.append(evaluate(rf_model,  X_test, y_test, "Random Forest"))

    # Feature importance instead of SHAP
    explain_features(xgb_model, X_test, features)

    # Save models
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost_severity.pkl")
    joblib.dump(rf_model,  f"{MODEL_DIR}/random_forest_severity.pkl")
    print(f"\n💾 Models saved to {MODEL_DIR}/")

    # Save results
    with open(f"{REPORTS_DIR}/model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "="*50)
    print("🏁 PHASE 5 COMPLETE!")
    print("="*50)

    for r in results:
        print(f"  {r['model']:20s} → Accuracy: {r['accuracy']} | F1: {r['f1_score']}")

    print("="*50)

if __name__ == "__main__":
    main()