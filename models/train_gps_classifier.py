import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "danishwasan-Ship-Navigation-System-Attacks-and-Defense-2738181/Dataset/submarine_navigation_data.csv"

df = pd.read_csv(DATA_PATH)
print("Loaded", len(df), "rows across scenarios:", df["Scenario"].unique())

def engineer_features(df):
    features = pd.DataFrame()
    features["gps_residual"] = np.sqrt(
        (df["Position_X"] - df["GPS_Position_X"])**2 +
        (df["Position_Y"] - df["GPS_Position_Y"])**2
    )
    features["fused_residual"] = np.sqrt(
        (df["Position_X"] - df["Fused_Position_X"])**2 +
        (df["Position_Y"] - df["Fused_Position_Y"])**2
    )
    features["velocity_mag"] = np.sqrt(df["Velocity_X"]**2 + df["Velocity_Y"]**2)
    features["course"] = df["Course"]
    features["depth"] = df["Depth"]
    return features

X_all = engineer_features(df)
y_all = df["Scenario"]

train_mask = y_all == "Normal"
X_train = X_all[train_mask]
print("Training Isolation Forest on", len(X_train), "normal rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
model.fit(X_train_scaled)

X_all_scaled = scaler.transform(X_all)
predictions = model.predict(X_all_scaled)  # 1 = normal, -1 = anomaly
scores = model.decision_function(X_all_scaled)

df["predicted_anomaly"] = predictions == -1
df["anomaly_score"] = scores

print("\n=== Detection rate per scenario ===")
summary = df.groupby("Scenario")["predicted_anomaly"].mean().sort_values(ascending=False)
print(summary)

print("\n=== Confusion: is_attack (true) vs predicted_anomaly ===")
df["is_attack"] = df["Scenario"] != "Normal"
print(confusion_matrix(df["is_attack"], df["predicted_anomaly"]))
print(classification_report(df["is_attack"], df["predicted_anomaly"], target_names=["Normal", "Attack"]))

joblib.dump(model, "models/isolation_forest_gps.joblib")
joblib.dump(scaler, "models/gps_feature_scaler.joblib")
print("\nModel saved to models/isolation_forest_gps.joblib")