import numpy as np
import pandas as pd
import joblib
import os
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Differential Privacy library (DP ML models)
from diffprivlib.models import (
    LinearRegression as DPLinearRegression,
    StandardScaler as DPStandardScaler
)

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# ---------------------------------------------------------
def load_and_preprocess(filepath):
    """
    Loads dataset and performs feature engineering:
    - Converts timestamps into numerical features
    - Encodes categorical variables
    - Creates vulnerability ratio
    - Removes unused columns
    """

    print("Loading dataset...")

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    header = lines[0].strip().strip('"')
    rows = [line.strip().strip('"') for line in lines[1:]]

    content = header + '\n' + '\n'.join(rows)
    df = pd.read_csv(StringIO(content), on_bad_lines='skip')

    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # ---------------- Timestamp feature engineering ----------------
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Minute"] = df["Timestamp"].dt.minute
    df["TimeInMinutes"] = df["Hour"] * 60 + df["Minute"]

    # ---------------- Encoding categorical variables ----------------
    zone_map = {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5, "Z6": 6}
    floor_map = {"Level 1": 1, "Level 2": 2}

    df["Zone_Enc"] = df["Zone_ID"].map(zone_map)
    df["Floor_Enc"] = df["Building_Level"].map(floor_map)

    # ---------------- Vulnerability feature ----------------
    df["Vulnerable_Ratio"] = (
        (df["Child_Count"] + df["Elderly_Count"] + df["Disabled_Count"])
        / df["Total_Occupancy_Raw"].replace(0, 1)
    )

    # ---------------- Binary encoding ----------------
    df["Exit_Passable_Enc"] = (df["Exit_Route_Passable"] == "Yes").astype(int)

    # ---------------- Drop unused columns ----------------
    df.drop(columns=[
        "Timestamp", "Incident_ID", "Building_Level", "Zone_ID",
        "Zone_Name", "Sprinkler_System_Status", "Hazard_Level_Rating",
        "Zone_Clearance_Status", "Exit_Route_Passable",
        "Last_Manual_Sweep_Time", "Event_Type", "Details"
    ], inplace=True)

    print(f"After preprocessing: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


# ---------------------------------------------------------
# 2. HAZARD SCORE MODEL (DIFFERENTIAL PRIVACY)
# ---------------------------------------------------------
def train_hazard_score_model_dp(df, epsilon=1.0):
    """
    Trains a Differentially Private Linear Regression model
    to predict hazard score.
    """

    print(f"Training DP Hazard Model (ε={epsilon})")

    features = [
        "PM2.5", "VOC", "Temperature", "CO2", "Humidity",
        "Hour", "TimeInMinutes", "Zone_Enc", "Floor_Enc",
        "Total_Occupancy_Raw", "Smoke_Obscuration_Pct",
        "Fire_Alarm_Triggered", "Vulnerable_Ratio"
    ]

    X = df[features]
    y = df["Hazard_Level_Score"]

    # Define feature and label bounds (required for DP models)
    bounds_X = (X.min().values, X.max().values)
    bounds_y = (y.min(), y.max())

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- Differential Privacy Scaling ----------------
    # Adds noise-aware scaling for privacy protection
    scaler = DPStandardScaler(bounds=bounds_X, epsilon=epsilon / 2)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------- DP Regression Model ----------------
    model = DPLinearRegression(
        bounds_X=bounds_X,
        bounds_y=bounds_y,
        epsilon=epsilon / 2
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict and clip negative values
    y_pred = np.clip(model.predict(X_test), 0, None)

    # ---------------- Evaluation ----------------
    print("=" * 45)
    print("MODEL 1 — DP Hazard Score")
    print("=" * 45)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R² : {r2_score(y_test, y_pred):.4f}\n")

    # Save trained model + scaler + feature list
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump((model, scaler, features), "saved_models/dp_hazard_score.pkl")

    return model, scaler, features


# ---------------------------------------------------------
# 3. EVACUATION PRIORITY MODEL (DIFFERENTIAL PRIVACY)
# ---------------------------------------------------------
def train_evacuation_model_dp(df, epsilon=1.0):
    """
    Trains a DP Linear Regression model to predict evacuation priority score.
    """

    print(f"Training DP Evacuation Model (ε={epsilon})")

    features = [
        "Adult_Count", "Child_Count", "Elderly_Count",
        "Disabled_Count", "Total_Occupancy_Raw",
        "Fire_Alarm_Triggered", "Smoke_Obscuration_Pct",
        "Temperature", "PM2.5", "VOC", "CO2", "Humidity",
        "Hour", "TimeInMinutes", "Zone_Enc", "Floor_Enc",
        "Vulnerable_Ratio", "Exit_Passable_Enc"
    ]

    X = df[features]
    y = df["Evacuation_Priority_Score"]

    bounds_X = (X.min().values, X.max().values)
    bounds_y = (y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # DP scaling step
    scaler = DPStandardScaler(bounds=bounds_X, epsilon=epsilon / 2)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # DP regression model
    model = DPLinearRegression(
        bounds_X=bounds_X,
        bounds_y=bounds_y,
        epsilon=epsilon / 2
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    print("=" * 45)
    print("MODEL 2 — DP Evacuation Score")
    print("=" * 45)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R² : {r2_score(y_test, y_pred):.4f}\n")

    # Save model
    joblib.dump((model, scaler, features), "saved_models/dp_evacuation_priority.pkl")

    return model, scaler, features


# ---------------------------------------------------------
# 4. DIFFERENTIAL PRIVACY POST-PROCESSING
# ---------------------------------------------------------
def apply_differential_privacy_updated(df, hazard_preds, evac_preds, epsilon=1.0):
    """
    Applies additional DP noise to sensitive outputs:
    - Occupancy (Laplace noise)
    - Heatmap (Gaussian noise)
    """

    print("=" * 45)
    print(f"Applying DP Noise (ε={epsilon})")
    print("=" * 45)

    # Laplace mechanism for occupancy (suitable for counts)
    def laplace_noise(series, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return series + np.random.laplace(0, scale, len(series))

    # Gaussian mechanism for continuous heatmap values
    def gaussian_noise(series, sensitivity, epsilon, delta=1e-5):
        sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        return series + np.random.normal(0, sigma, len(series))

    # DP-protected occupancy values
    dp_occupancy = laplace_noise(
        df["Total_Occupancy_Raw"].astype(float),
        1.0, epsilon
    ).clip(0).round().astype(int)

    # Normalize and apply Gaussian noise for heatmap
    heatmap_raw = df["Total_Occupancy_Raw"] / df["Total_Occupancy_Raw"].max()
    dp_heatmap = gaussian_noise(heatmap_raw, 1.0, epsilon).clip(0, 1)

    # Final dashboard dataset
    dashboard_df = pd.DataFrame({
        "Zone_ID": df["Zone_Enc"].values,
        "DP_Occupancy": dp_occupancy.values,
        "DP_Hazard_Score": np.round(hazard_preds, 3),
        "DP_Evac_Score": np.round(evac_preds, 2),
        "DP_Heatmap": np.round(dp_heatmap, 3)
    })

    print("✅ Dashboard-ready DP output generated")
    return dashboard_df


# ---------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------
if __name__ == "__main__":

    DATASET_PATH = "new_dataset.csv"

    # Load and preprocess data
    df = load_and_preprocess(DATASET_PATH)

    # Train DP models
    hazard_model, hazard_scaler, hazard_features = train_hazard_score_model_dp(df)
    evac_model, evac_scaler, evac_features = train_evacuation_model_dp(df)

    # Generate DP predictions (apply scaling during inference)
    hazard_preds = hazard_model.predict(hazard_scaler.transform(df[hazard_features]))
    evac_preds = evac_model.predict(evac_scaler.transform(df[evac_features]))

    # Apply output-level differential privacy
    dashboard_output = apply_differential_privacy_updated(
        df,
        hazard_preds,
        evac_preds,
        epsilon=1.0
    )

    print("=" * 45)
    print("Privacy-First Pipeline Complete")
    print("=" * 45)

    dashboard_output.to_csv("dp_output.csv", index=False)
    print("✅ dp_output.csv saved successfully!")