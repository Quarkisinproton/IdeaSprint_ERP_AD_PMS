import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
import shap
import joblib

# Create models directory
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("  ERR-CC Data Engine v2.0 (with SHAP Explainability)")
print("=" * 60)

print("\n[Phase 1] Data Processing & Feature Engineering")

# ==========================================
# 1. Financial Data (PaySim) Processing
# ==========================================
print("  → Loading PaySim Data...")
df_paysim = pd.read_csv("DataSets/PAYSim/paysim.csv", nrows=100000)

df_paysim['amount'] = pd.to_numeric(df_paysim['amount'], errors='coerce').fillna(0)
df_paysim['type'] = df_paysim['type'].astype(str)

# Create 'balance_error' feature
df_paysim['balance_error'] = (df_paysim['oldbalanceOrg'] - df_paysim['amount']) - df_paysim['newbalanceOrig']

# ==========================================
# 2. Maintenance Data (AI4I) Processing
# ==========================================
print("  → Loading AI4I Data...")
df_ai4i = pd.read_csv("DataSets/AI4I/ai4i2020.csv")

scaler = StandardScaler()
df_ai4i[['Tool wear [min]', 'Torque [Nm]']] = scaler.fit_transform(df_ai4i[['Tool wear [min]', 'Torque [Nm]']])

# ==========================================
# 3. NASA Data Processing (Synthetic)
# ==========================================
print("  → Generating NASA C-MAPSS Synthetic Data...")
np.random.seed(42)
engine_ids = np.repeat(np.arange(1, 11), 50)
cycles = np.tile(np.arange(1, 51), 10)
# Generate multiple correlated sensors for richer SHAP explanations
sensor_1 = 100 + np.random.normal(0, 10, 500) - (cycles * 0.3)
sensor_2 = 200 + np.random.normal(0, 5, 500) + (cycles * 0.15)
sensor_3 = 50 + np.random.normal(0, 8, 500) - (cycles * 0.2)

df_nasa = pd.DataFrame({
    'EngineID': engine_ids, 'Cycle': cycles,
    'Sensor_1': sensor_1, 'Sensor_2': sensor_2, 'Sensor_3': sensor_3
})

max_cycles = df_nasa.groupby('EngineID')['Cycle'].max().reset_index()
max_cycles.rename(columns={'Cycle': 'Max_Cycle'}, inplace=True)
df_nasa = df_nasa.merge(max_cycles, on='EngineID')
df_nasa['RUL'] = df_nasa['Max_Cycle'] - df_nasa['Cycle']

print("  ✓ Phase 1 Complete.\n")

# ==========================================
# Phase 2: Model Training
# ==========================================
print("[Phase 2] Model Training")

# --- Isolation Forest (Financial Fraud) ---
print("  → Training Isolation Forest on PaySim...")
fraud_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error']
X_fraud = df_paysim[fraud_features]

iforest = IForest(contamination=0.01, random_state=42)
iforest.fit(X_fraud)
joblib.dump(iforest, "models/iforest_fraud.pkl")
print("    Saved iforest_fraud.pkl")

df_paysim['Risk_Score'] = iforest.decision_scores_
df_paysim['Is_Anomaly'] = iforest.labels_
df_paysim.to_csv("DataSets/PAYSim/processed_paysim.csv", index=False)

# --- Random Forest Regressor (RUL) ---
print("  → Training Random Forest Regressor for RUL...")
rul_features = ['Cycle', 'Sensor_1', 'Sensor_2', 'Sensor_3']
X_rul = df_nasa[rul_features]
y_rul = df_nasa['RUL']

rf_rul = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rul.fit(X_rul, y_rul)
joblib.dump(rf_rul, "models/rf_rul.pkl")
print("    Saved rf_rul.pkl")

df_nasa.to_csv("DataSets/Synthetic/processed_nasa.csv", index=False)
print("  ✓ Phase 2 Complete.\n")

# ==========================================
# Phase 3: SHAP Explainability
# ==========================================
print("[Phase 3] Computing SHAP Values for Explainability")

# --- SHAP for RUL Model (TreeExplainer - fast) ---
print("  → Computing SHAP for RUL Model (TreeExplainer)...")
explainer_rul = shap.TreeExplainer(rf_rul)
shap_values_rul = explainer_rul.shap_values(X_rul)
joblib.dump(shap_values_rul, "models/shap_values_rul.pkl")
joblib.dump(explainer_rul.expected_value, "models/shap_expected_rul.pkl")
print("    Saved shap_values_rul.pkl")

# --- SHAP for Fraud Model (using sklearn IsolationForest wrapper for TreeExplainer) ---
print("  → Computing SHAP for Fraud Model (TreeExplainer on IsolationForest)...")
# Train a sklearn IsolationForest for SHAP compatibility (PyOD wraps sklearn internally)
from sklearn.ensemble import IsolationForest as SklearnIF
sklearn_iforest = SklearnIF(contamination=0.01, random_state=42)
sklearn_iforest.fit(X_fraud)
joblib.dump(sklearn_iforest, "models/sklearn_iforest_fraud.pkl")

# Use a sample for SHAP to keep it fast
fraud_sample = X_fraud.sample(n=min(500, len(X_fraud)), random_state=42)
explainer_fraud = shap.TreeExplainer(sklearn_iforest)
shap_values_fraud = explainer_fraud.shap_values(fraud_sample)
joblib.dump(shap_values_fraud, "models/shap_values_fraud.pkl")
joblib.dump(fraud_sample, "models/shap_fraud_sample.pkl")
joblib.dump(explainer_fraud.expected_value, "models/shap_expected_fraud.pkl")
print("    Saved shap_values_fraud.pkl")

print("  ✓ Phase 3 Complete.\n")
print("=" * 60)
print("  All phases complete. Models and SHAP artifacts saved to /models.")
print("=" * 60)
