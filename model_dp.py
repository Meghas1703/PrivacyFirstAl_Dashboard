import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

DATASET_PATH = "data/emergency_zone_data.csv"
CLUSTER_PATH = "data/occupancy_clusters.csv"
OUTPUT_PATH  = "data/dashboard_model_output.csv"
HEATMAP_PATH = "data/dp_dashboard_heatmap.csv"
MODEL_DIR    = "saved_models"


# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# ---------------------------------------------------------

def load_dataset(filepath=DATASET_PATH):
    print("Loading dataset...")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Time features
    df["Hour"]          = df["Timestamp"].dt.hour
    df["Minute"]        = df["Timestamp"].dt.minute
    df["TimeInMinutes"] = df["Hour"] * 60 + df["Minute"]

    # Encode zone and floor
    zone_map  = {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5, "Z6": 6}
    floor_map = {"Level 1": 1, "Level 2": 2}
    df["Zone_Enc"]  = df["Zone_ID"].map(zone_map)
    df["Floor_Enc"] = df["Building_Level"].map(floor_map)

    # Vulnerability features
    df["Vulnerable_Count"] = (
        df["Child_Count"] + df["Elderly_Count"] + df["Disabled_Count"]
    )
    df["Vulnerable_Ratio"] = (
        df["Vulnerable_Count"] / df["Total_Occupancy_Raw"].replace(0, 1)
    )

    # Encode route passability
    df["Exit_Passable_Enc"] = df["Exit_Route_Passable"].map({
        "Yes": 1.0, "Partial": 0.5, "No": 0.0,
    }).fillna(0.0)

    # Encode sprinkler status
    if "Sprinkler_System_Status" in df.columns:
        df["Sprinkler_Enc"] = df["Sprinkler_System_Status"].map({
            "Off": 0.0, "On": 1.0, "Activated": 1.0,
        }).fillna(0.0)
    else:
        df["Sprinkler_Enc"] = 0.0

    # Rate-of-change trends — catch a spreading fire earlier than static values alone
    df = df.sort_values(["Zone_ID", "Timestamp"])
    df["Temp_Trend"]  = df.groupby("Zone_ID")["Temperature"].diff().fillna(0)
    df["Smoke_Trend"] = df.groupby("Zone_ID")["Smoke_Obscuration_Pct"].diff().fillna(0)
    df["VOC_Trend"]   = df.groupby("Zone_ID")["VOC"].diff().fillna(0)

    # Weighted vulnerability — disabled need more evacuation time than children
    df["Vulnerable_Weighted"] = (
        df["Child_Count"]    * 1.3
        + df["Elderly_Count"]  * 1.5
        + df["Disabled_Count"] * 1.9
    ) / df["Total_Occupancy_Raw"].replace(0, 1)

    # --- HIGH-SIGNAL FEATURES (from friend's approach) ---
    # Log transforms shrink the range of skewed sensor data so tree splits are sharper
    df["PM25_Log"] = np.log1p(df["PM2.5"])
    df["VOC_Log"]  = np.log1p(df["VOC"])

    # Binary fire signal — direct, noise-resistant indicator
    df["Fire_Indicator_Binary"] = (
        (df["Temperature"] > 32) & (df["Smoke_Obscuration_Pct"] > 18)
    ).astype(int)

    # Combined environmental risk index scaled 0–1
    df["Env_Risk_Index"] = (df["PM25_Log"] * df["Temperature"]) / 250.0

    # Interaction feature — hot + smoky together is more dangerous than either alone
    df["Smoke_Temp_Interaction"] = df["Temperature"] * df["Smoke_Obscuration_Pct"]

    df = df.fillna(0)

    print("Dataset loaded and feature engineering completed.")
    print("-" * 60)

    return df


# ---------------------------------------------------------
# 2. TRAIN MODEL (accurate, non-DP — DP is applied at display layer)
# ---------------------------------------------------------

def train_model(df, features, target, model, model_name):
    print(f"Training {model_name}...")

    X = df[features].copy().fillna(0)
    y = df[target].copy().fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"model": model, "features": features, "target": target, "mae": mae, "r2": r2},
        f"{MODEL_DIR}/{model_name}.pkl",
    )

    print("=" * 60)
    print(model_name)
    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}")
    print("=" * 60)
    print()

    return model, features, mae, r2


# ---------------------------------------------------------
# 3. EPSILON SENSITIVITY AUDIT
# ---------------------------------------------------------

def run_epsilon_audit(hazard_mae, hazard_r2, evac_mae, evac_r2):
    """
    Shows the privacy-vs-noise tradeoff for the display layer.

    In the hybrid approach, model accuracy is constant (no DP in training).
    Epsilon only controls how much noise is added to the displayed counts
    and heatmap coordinates.
    """

    print("\n" + "=" * 60)
    print("EPSILON SENSITIVITY AUDIT — DISPLAY LAYER")
    print("=" * 60)
    print(f"Model accuracy is independent of epsilon in the hybrid approach.")
    print(f"  Hazard Model     — MAE: {hazard_mae:.4f} | R2: {hazard_r2:.4f}")
    print(f"  Evacuation Model — MAE: {evac_mae:.4f} | R2: {evac_r2:.4f}")
    print()

    results = []

    for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
        count_margin = round((1.0 / eps) * np.log(20), 1)
        coord_margin = round((2.0 / eps) * np.log(20), 1)
        results.append({
            "Epsilon": eps,
            "Privacy Level":    "Strong" if eps <= 0.5 else ("Balanced" if eps <= 2.0 else "Light"),
            "Count Noise (±)":  f"±{count_margin} persons",
            "Coord Noise (±)":  f"±{coord_margin} m",
        })

    print(pd.DataFrame(results).to_string(index=False))
    print()
    print("Production epsilon = 1.0 (balanced — ±3 persons, ±6m coord blur)")
    print("=" * 60)
    print()

    # MODEL PERFORMANCE COMPARISON TABLE
    print("📊 MODEL PERFORMANCE COMPARISON (PRIVACY VS UTILITY)")
    print("=" * 60)

    perf_results = []
    for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
        warden_error = round((1.0 / eps) * np.log(20), 1)
        perf_results.append({
            "Epsilon":       eps,
            "MAE":           round(hazard_mae, 3),
            "R2":            round(hazard_r2,  4),
            "Warden_Error":  f"±{warden_error}",
        })

    print(pd.DataFrame(perf_results).to_string(index=False))
    print("=" * 60)
    print()


# ---------------------------------------------------------
# 4. DIFFERENTIAL PRIVACY LAYER (DISPLAY COUNTS)
# ---------------------------------------------------------

def apply_laplace_noise(series, epsilon=1.0, sensitivity=1.0):
    """
    Applies Laplace noise to count values for dashboard display.
    Lower epsilon = stronger privacy, more noise.
    """

    epsilon = max(float(epsilon), 0.01)
    scale   = sensitivity / epsilon
    noisy   = series.astype(float) + np.random.laplace(0, scale, len(series))

    return noisy.clip(0).round().astype(int)


# ---------------------------------------------------------
# 5. HEATMAP BLUR OUTPUT
# ---------------------------------------------------------

def create_heatmap_output(cluster_df, epsilon=1.0):
    """
    Blurs cluster coordinates and sizes before saving.
    Sensitivity=2.0 gives roughly ±6m spatial blur at epsilon=1.0.
    """

    print("Creating DP heatmap output...")

    out   = cluster_df.copy()
    scale = 2.0 / max(float(epsilon), 0.01)

    out["DP_Coord_X"]      = (cluster_df["Coord_X"] + np.random.laplace(0, scale, len(cluster_df))).round(2)
    out["DP_Coord_Y"]      = (cluster_df["Coord_Y"] + np.random.laplace(0, scale, len(cluster_df))).round(2)
    out["DP_Cluster_Size"] = apply_laplace_noise(cluster_df["Cluster_Size"], epsilon=epsilon)

    out.to_csv(HEATMAP_PATH, index=False)

    print(f"Heatmap output saved to: {HEATMAP_PATH}")
    print("-" * 60)

    return out


# ---------------------------------------------------------
# 6. CREATE DASHBOARD OUTPUT
# ---------------------------------------------------------

def create_dashboard_output(
    df,
    hazard_model,
    hazard_features,
    evac_model,
    evac_features,
    epsilon=1.0,
):
    """
    Creates dashboard-ready dataset in the same format app.py expects.

    Adds:
    - Predicted hazard score + label
    - Predicted evacuation priority score + label
    - DP-protected occupancy and vulnerable population counts
    - DP heatmap value
    """

    print("Creating dashboard-ready output...")

    df_out = df.copy()

    hazard_preds = np.clip(hazard_model.predict(df_out[hazard_features].fillna(0)), 0, None)
    evac_preds   = np.clip(evac_model.predict(df_out[evac_features].fillna(0)),   0, None)

    df_out["Predicted_Hazard_Score"]              = hazard_preds.round(2)
    df_out["Predicted_Evacuation_Priority_Score"] = evac_preds.round(2)

    # DP-protected counts for dashboard display
    df_out["DP_Total_Occupancy"]  = apply_laplace_noise(df_out["Total_Occupancy_Raw"], epsilon=epsilon)
    df_out["DP_Adult_Count"]      = apply_laplace_noise(df_out["Adult_Count"],         epsilon=epsilon)
    df_out["DP_Child_Count"]      = apply_laplace_noise(df_out["Child_Count"],         epsilon=epsilon)
    df_out["DP_Elderly_Count"]    = apply_laplace_noise(df_out["Elderly_Count"],       epsilon=epsilon)
    df_out["DP_Disabled_Count"]   = apply_laplace_noise(df_out["Disabled_Count"],      epsilon=epsilon)
    df_out["DP_Vulnerable_Count"] = (
        df_out["DP_Child_Count"] + df_out["DP_Elderly_Count"] + df_out["DP_Disabled_Count"]
    )

    max_occ = df_out["DP_Total_Occupancy"].max()
    df_out["DP_Heatmap_Value"] = (
        (df_out["DP_Total_Occupancy"] / max_occ).round(3) if max_occ > 0 else 0.0
    )

    # Human-readable labels
    df_out["Predicted_Hazard_Level"] = pd.cut(
        df_out["Predicted_Hazard_Score"],
        bins=[-1, 25, 50, 75, float("inf")],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)

    df_out["Predicted_Evacuation_Priority_Level"] = pd.cut(
        df_out["Predicted_Evacuation_Priority_Score"],
        bins=[-1, 25, 50, 75, float("inf")],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)

    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Dashboard output saved to: {OUTPUT_PATH}")
    print("-" * 60)

    return df_out


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------

def main(epsilon=1.0):
    df         = load_dataset(DATASET_PATH)
    cluster_df = pd.read_csv(CLUSTER_PATH)
    cluster_df["Timestamp"] = pd.to_datetime(cluster_df["Timestamp"])

    # Enhanced hazard features — original set + high-signal additions
    hazard_features = [
        "PM25_Log",
        "VOC_Log",
        "Temperature",
        "CO2",
        "Humidity",
        "Hour",
        "TimeInMinutes",
        "Zone_Enc",
        "Floor_Enc",
        "Total_Occupancy_Raw",
        "Smoke_Obscuration_Pct",
        "Fire_Alarm_Triggered",
        "Vulnerable_Ratio",
        "Sprinkler_Enc",
        "Fire_Indicator_Binary",
        "Env_Risk_Index",
        "Smoke_Temp_Interaction",
        "Temp_Trend",
        "Smoke_Trend",
        "VOC_Trend",
    ]

    hazard_model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    hazard_model, hazard_features, hazard_mae, hazard_r2 = train_model(
        df=df,
        features=hazard_features,
        target="Hazard_Level_Score",
        model=hazard_model,
        model_name="gradient_boosting_hazard_score_model",
    )

    # Enhanced evacuation features — original set + high-signal additions
    evacuation_features = [
        "Adult_Count",
        "Child_Count",
        "Elderly_Count",
        "Disabled_Count",
        "Total_Occupancy_Raw",
        "Vulnerable_Count",
        "Vulnerable_Ratio",
        "Fire_Alarm_Triggered",
        "Smoke_Obscuration_Pct",
        "Temperature",
        "PM25_Log",
        "VOC_Log",
        "CO2",
        "Humidity",
        "Hour",
        "TimeInMinutes",
        "Zone_Enc",
        "Floor_Enc",
        "Exit_Passable_Enc",
        "Sprinkler_Enc",
        "Fire_Indicator_Binary",
        "Env_Risk_Index",
        "Smoke_Temp_Interaction",
        "Temp_Trend",
        "Smoke_Trend",
        "VOC_Trend",
        "Vulnerable_Weighted",
    ]

    evac_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    evac_model, evacuation_features, evac_mae, evac_r2 = train_model(
        df=df,
        features=evacuation_features,
        target="Evacuation_Priority_Score",
        model=evac_model,
        model_name="random_forest_evacuation_priority_model",
    )

    # Epsilon audit — shows display-layer privacy tradeoff
    run_epsilon_audit(hazard_mae, hazard_r2, evac_mae, evac_r2)

    # Heatmap blur for cluster coordinates
    create_heatmap_output(cluster_df, epsilon=epsilon)

    # Dashboard CSV (same format app.py reads)
    create_dashboard_output(
        df=df,
        hazard_model=hazard_model,
        hazard_features=hazard_features,
        evac_model=evac_model,
        evac_features=evacuation_features,
        epsilon=epsilon,
    )

    print("Pipeline completed successfully.")
    print(f"Hazard Model     — MAE: {hazard_mae:.4f} | R2: {hazard_r2:.4f}")
    print(f"Evacuation Model — MAE: {evac_mae:.4f} | R2: {evac_r2:.4f}")


if __name__ == "__main__":
    main(epsilon=1.0)
