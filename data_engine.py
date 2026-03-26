"""
============================================================
  ERR-CC Data Engine v3.0 — Enterprise Risk & Reliability
  Unified Risk Intelligence Pipeline
============================================================
Features:
- Ensemble Anomaly Detection (IForest + LOF + XGBoost)
- Graph-Based Fraud Network Analysis (NetworkX)
- Advanced RUL with XGBoost + Digital Twin features
- Credit Card Fraud integration
- SHAP Explainability for all models
- Cross-domain risk correlation
============================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import xgboost as xgb
import shap
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')
os.makedirs("models", exist_ok=True)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     ERR-CC DATA ENGINE v3.0 — Unified Risk Intelligence     ║
║     Enterprise Risk & Reliability Command Center             ║
╚══════════════════════════════════════════════════════════════╝
"""
print(BANNER)

# ============================================================
# PHASE 1: DATA INGESTION & FEATURE ENGINEERING
# ============================================================
print("━" * 60)
print("PHASE 1: Data Ingestion & Feature Engineering")
print("━" * 60)

# --- 1A. PaySim Financial Data ---
print("\n  [1/4] Loading PaySim Financial Data...")
df_paysim = pd.read_csv("DataSets/PAYSim/paysim.csv", nrows=150000)
df_paysim['amount'] = pd.to_numeric(df_paysim['amount'], errors='coerce').fillna(0)
df_paysim['type'] = df_paysim['type'].astype(str)

# Feature Engineering
df_paysim['balance_error_orig'] = (df_paysim['oldbalanceOrg'] - df_paysim['amount']) - df_paysim['newbalanceOrig']
df_paysim['balance_error_dest'] = (df_paysim['oldbalanceDest'] + df_paysim['amount']) - df_paysim['newbalanceDest']
df_paysim['amount_to_balance_ratio'] = df_paysim['amount'] / (df_paysim['oldbalanceOrg'] + 1)
df_paysim['is_zero_balance_after'] = (df_paysim['newbalanceOrig'] == 0).astype(int)
df_paysim['is_full_transfer'] = ((df_paysim['amount'] == df_paysim['oldbalanceOrg']) & (df_paysim['oldbalanceOrg'] > 0)).astype(int)

# Encode transaction type
le_type = LabelEncoder()
df_paysim['type_encoded'] = le_type.fit_transform(df_paysim['type'])
joblib.dump(le_type, "models/label_encoder_type.pkl")

print(f"    → {len(df_paysim):,} transactions loaded")
print(f"    → Fraud rate: {df_paysim['isFraud'].mean()*100:.3f}%")

# --- 1B. Credit Card Fraud Data ---
print("\n  [2/4] Loading Credit Card Fraud Data...")
df_cc = pd.read_csv("DataSets/Credit Card Fraud/credit_card_fraud_10k.csv")
print(f"    → {len(df_cc):,} credit card transactions loaded")
print(f"    → Fraud rate: {df_cc['is_fraud'].mean()*100:.2f}%")

# --- 1C. AI4I Predictive Maintenance ---
print("\n  [3/4] Loading AI4I Maintenance Data...")
df_ai4i = pd.read_csv("DataSets/AI4I/ai4i2020.csv")
scaler_ai4i = StandardScaler()
df_ai4i[['Tool wear [min]', 'Torque [Nm]']] = scaler_ai4i.fit_transform(
    df_ai4i[['Tool wear [min]', 'Torque [Nm]']]
)
joblib.dump(scaler_ai4i, "models/scaler_ai4i.pkl")
print(f"    → {len(df_ai4i):,} machine records loaded")

# --- 1D. NASA C-MAPSS Synthetic (Enhanced for Digital Twin) ---
print("\n  [4/4] Generating NASA C-MAPSS Enhanced Synthetic Data...")
np.random.seed(42)
n_engines = 20
max_life_range = (120, 250)

records = []
for eid in range(1, n_engines + 1):
    max_life = np.random.randint(*max_life_range)
    for cycle in range(1, max_life + 1):
        progress = cycle / max_life
        # Sensors degrade as engine approaches failure
        s1 = 518.67 + np.random.normal(0, 2) - (progress ** 2) * 30   # Total temp
        s2 = 642.15 + np.random.normal(0, 3) + (progress ** 1.5) * 20  # LPC outlet temp
        s3 = 1580 + np.random.normal(0, 5) - (progress ** 2) * 50      # HPC outlet temp
        s4 = 47.47 + np.random.normal(0, 0.5) + progress * 3           # LPT outlet temp
        s5 = 521.66 + np.random.normal(0, 1) - (progress ** 1.8) * 15  # Bypass ratio
        # Operational settings
        op1 = np.random.choice([-0.0007, 0.0, 0.001, 0.0042])
        op2 = np.random.choice([0.0001, 0.0003, 0.0007])
        records.append({
            'EngineID': eid, 'Cycle': cycle, 'Max_Cycle': max_life,
            'OpSetting_1': op1, 'OpSetting_2': op2,
            'Sensor_1_TotalTemp': round(s1, 2),
            'Sensor_2_LPC': round(s2, 2),
            'Sensor_3_HPC': round(s3, 2),
            'Sensor_4_LPT': round(s4, 2),
            'Sensor_5_Bypass': round(s5, 2),
        })

df_nasa = pd.DataFrame(records)
df_nasa['RUL'] = df_nasa['Max_Cycle'] - df_nasa['Cycle']
print(f"    → {len(df_nasa):,} engine telemetry records generated")
print(f"    → {n_engines} engines, life range: {max_life_range}")

print("\n  ✓ Phase 1 Complete.")

# ============================================================
# PHASE 2: GRAPH-BASED FRAUD NETWORK ANALYSIS
# ============================================================
print("\n" + "━" * 60)
print("PHASE 2: Graph-Based Fraud Network Analysis")
print("━" * 60)

print("  → Building transaction graph (sender → receiver)...")
# Build directed graph from PaySim
G = nx.DiGraph()
# Use a sample for graph analysis (performance)
graph_sample = df_paysim[df_paysim['type'].isin(['TRANSFER', 'CASH_OUT'])].head(30000)

for _, row in graph_sample.iterrows():
    G.add_edge(row['nameOrig'], row['nameDest'], weight=row['amount'], type=row['type'])

print(f"    → Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Detect circular payment patterns (cycles in graph)
print("  → Detecting circular payment patterns...")
cycles_found = []
try:
    for cycle in nx.simple_cycles(G):
        if 2 <= len(cycle) <= 5:
            cycles_found.append(cycle)
        if len(cycles_found) >= 50:
            break
except Exception:
    pass
print(f"    → {len(cycles_found)} circular payment rings detected")

# Compute node risk metrics
print("  → Computing node-level risk metrics...")
pagerank = nx.pagerank(G, weight='weight', max_iter=50)
in_degree = dict(G.in_degree(weight='weight'))
out_degree = dict(G.out_degree(weight='weight'))

# Flag high-risk nodes (accounts appearing in cycles or with high PageRank)
cycle_nodes = set()
for c in cycles_found:
    cycle_nodes.update(c)

graph_risk_data = []
for node in G.nodes():
    graph_risk_data.append({
        'account': node,
        'pagerank': pagerank.get(node, 0),
        'in_flow': in_degree.get(node, 0),
        'out_flow': out_degree.get(node, 0),
        'in_circular_ring': node in cycle_nodes,
        'connections': G.degree(node),
    })
df_graph_risk = pd.DataFrame(graph_risk_data)
df_graph_risk['flow_ratio'] = df_graph_risk['out_flow'] / (df_graph_risk['in_flow'] + 1)
df_graph_risk.to_csv("DataSets/Synthetic/graph_risk_nodes.csv", index=False)

# Save graph data for visualization
graph_edges = []
for u, v, data in G.edges(data=True):
    graph_edges.append({'source': u, 'target': v, 'weight': data['weight'], 'type': data['type']})
df_graph_edges = pd.DataFrame(graph_edges)
# Save top suspicious edges (high weight or connected to cycle nodes)
suspicious_edges = df_graph_edges[
    (df_graph_edges['source'].isin(cycle_nodes)) | 
    (df_graph_edges['target'].isin(cycle_nodes))
].head(500)
suspicious_edges.to_csv("DataSets/Synthetic/graph_suspicious_edges.csv", index=False)

# Save cycles for UI
cycles_export = [{'ring_id': i, 'accounts': c, 'size': len(c)} for i, c in enumerate(cycles_found)]
with open("models/fraud_rings.json", "w") as f:
    json.dump(cycles_export, f)

print("  ✓ Phase 2 Complete.")

# ============================================================
# PHASE 3: ENSEMBLE ANOMALY DETECTION (Financial)
# ============================================================
print("\n" + "━" * 60)
print("PHASE 3: Ensemble Anomaly Detection")
print("━" * 60)

fraud_features = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'balance_error_orig', 'balance_error_dest',
    'amount_to_balance_ratio', 'is_zero_balance_after',
    'is_full_transfer', 'type_encoded'
]
X_fraud = df_paysim[fraud_features].copy()

# Replace inf values
X_fraud.replace([np.inf, -np.inf], 0, inplace=True)
X_fraud.fillna(0, inplace=True)

# --- Model 1: Isolation Forest (PyOD) ---
print("  → Training Isolation Forest...")
iforest = IForest(contamination=0.01, n_estimators=150, random_state=42)
iforest.fit(X_fraud)

# --- Model 2: Local Outlier Factor (PyOD) ---
print("  → Training Local Outlier Factor...")
lof = LOF(contamination=0.01, n_neighbors=20)
lof.fit(X_fraud)

# --- Model 3: XGBoost Classifier (Credit Card Data as supervised signal) ---
print("  → Training XGBoost Fraud Classifier on Credit Card data...")
cc_features = ['amount', 'transaction_hour', 'foreign_transaction', 
               'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']
X_cc = df_cc[cc_features]
y_cc = df_cc['is_fraud']

# Handle class imbalance with scale_pos_weight
fraud_ratio = (y_cc == 0).sum() / max((y_cc == 1).sum(), 1)
xgb_fraud = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=fraud_ratio, eval_metric='aucpr',
    random_state=42, use_label_encoder=False
)
xgb_fraud.fit(X_cc, y_cc)

# --- Ensemble Scoring for PaySim ---
print("  → Computing Ensemble Risk Scores...")
scores_iforest = iforest.decision_scores_
scores_lof = lof.decision_scores_

# Normalize scores to [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler_scores = MinMaxScaler()
ensemble_scores = scaler_scores.fit_transform(
    np.column_stack([scores_iforest, scores_lof])
)
df_paysim['Risk_Score_IForest'] = ensemble_scores[:, 0]
df_paysim['Risk_Score_LOF'] = ensemble_scores[:, 1]
df_paysim['Risk_Score_Ensemble'] = (ensemble_scores[:, 0] * 0.5 + ensemble_scores[:, 1] * 0.5)

# Threshold: top 1%
threshold = np.percentile(df_paysim['Risk_Score_Ensemble'], 99)
df_paysim['Is_Anomaly'] = (df_paysim['Risk_Score_Ensemble'] >= threshold).astype(int)

# Add graph-based risk
df_paysim = df_paysim.merge(
    df_graph_risk[['account', 'pagerank', 'in_circular_ring']].rename(columns={'account': 'nameOrig'}),
    on='nameOrig', how='left'
)
df_paysim['pagerank'] = df_paysim['pagerank'].fillna(0)
df_paysim['in_circular_ring'] = df_paysim['in_circular_ring'].fillna(False)

# Save everything
joblib.dump(iforest, "models/iforest_fraud.pkl")
joblib.dump(lof, "models/lof_fraud.pkl")
joblib.dump(xgb_fraud, "models/xgb_fraud.pkl")
df_paysim.to_csv("DataSets/PAYSim/processed_paysim.csv", index=False)
df_cc.to_csv("DataSets/Credit Card Fraud/processed_cc.csv", index=False)

print(f"    → Flagged {df_paysim['Is_Anomaly'].sum():,} anomalous transactions")
print("  ✓ Phase 3 Complete.")

# ============================================================
# PHASE 4: ADVANCED RUL PREDICTION (XGBoost + Digital Twin)
# ============================================================
print("\n" + "━" * 60)
print("PHASE 4: Advanced RUL Prediction & Digital Twin")
print("━" * 60)

sensor_cols = [c for c in df_nasa.columns if c.startswith('Sensor_')]
rul_features = ['Cycle', 'OpSetting_1', 'OpSetting_2'] + sensor_cols
X_rul = df_nasa[rul_features]
y_rul = df_nasa['RUL']

# XGBoost Regressor for RUL
print("  → Training XGBoost RUL Regressor...")
xgb_rul = xgb.XGBRegressor(
    n_estimators=200, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42
)
xgb_rul.fit(X_rul, y_rul)
joblib.dump(xgb_rul, "models/xgb_rul.pkl")

# Also train RF for ensemble
print("  → Training Random Forest RUL Regressor (ensemble)...")
from sklearn.ensemble import RandomForestRegressor
rf_rul = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rul.fit(X_rul, y_rul)
joblib.dump(rf_rul, "models/rf_rul.pkl")

# Save processed NASA data
df_nasa.to_csv("DataSets/Synthetic/processed_nasa.csv", index=False)

# Save feature list
joblib.dump(rul_features, "models/rul_features.pkl")
joblib.dump(sensor_cols, "models/sensor_cols.pkl")

print("  ✓ Phase 4 Complete.")

# ============================================================
# PHASE 5: SHAP EXPLAINABILITY
# ============================================================
print("\n" + "━" * 60)
print("PHASE 5: SHAP Explainability Engine")
print("━" * 60)

# --- SHAP for XGBoost RUL ---
print("  → SHAP: XGBoost RUL Model...")
explainer_rul = shap.TreeExplainer(xgb_rul)
# Use a sample for speed
rul_sample_idx = np.random.choice(len(X_rul), size=min(1000, len(X_rul)), replace=False)
X_rul_sample = X_rul.iloc[rul_sample_idx]
shap_values_rul = explainer_rul.shap_values(X_rul_sample)
joblib.dump(shap_values_rul, "models/shap_values_rul.pkl")
joblib.dump(X_rul_sample, "models/shap_rul_sample.pkl")
joblib.dump(rul_sample_idx, "models/shap_rul_sample_idx.pkl")
joblib.dump(explainer_rul.expected_value, "models/shap_expected_rul.pkl")

# --- SHAP for XGBoost Fraud Classifier ---
print("  → SHAP: XGBoost Fraud Classifier...")
explainer_xgb_fraud = shap.TreeExplainer(xgb_fraud)
shap_values_cc = explainer_xgb_fraud.shap_values(X_cc)
joblib.dump(shap_values_cc, "models/shap_values_cc.pkl")
joblib.dump(explainer_xgb_fraud.expected_value, "models/shap_expected_cc.pkl")

# --- SHAP for Isolation Forest ---
print("  → SHAP: Isolation Forest (sklearn wrapper)...")
sklearn_if = IsolationForest(contamination=0.01, n_estimators=150, random_state=42)
sklearn_if.fit(X_fraud)
fraud_sample = X_fraud.sample(n=min(500, len(X_fraud)), random_state=42)
explainer_if = shap.TreeExplainer(sklearn_if)
shap_values_if = explainer_if.shap_values(fraud_sample)
joblib.dump(shap_values_if, "models/shap_values_iforest.pkl")
joblib.dump(fraud_sample, "models/shap_fraud_sample.pkl")
joblib.dump(sklearn_if, "models/sklearn_iforest.pkl")

print("  ✓ Phase 5 Complete.")

# ============================================================
# PHASE 6: CROSS-DOMAIN RISK CORRELATION
# ============================================================
print("\n" + "━" * 60)
print("PHASE 6: Cross-Domain Risk Intelligence")
print("━" * 60)

# Create a mock "Vendor Database" linking engines to procurement costs
print("  → Building Vendor & Procurement Knowledge Base...")
vendor_db = []
part_names = ['Turbine Blade Assembly', 'HPC Compressor Ring', 'Bearing Housing Unit',
              'Fuel Nozzle Kit', 'Combustion Liner', 'LPT Vane Segment']
vendors = ['AeroTech Corp', 'Global Turbines Ltd', 'PrecisionAero Inc', 'JetParts International']

for eid in range(1, n_engines + 1):
    part = np.random.choice(part_names)
    vendor = np.random.choice(vendors)
    base_cost = np.random.uniform(15000, 85000)
    # Some vendors engage in "price gouging" (30% chance)
    is_gouging = np.random.random() < 0.3
    quoted_price = base_cost * (1.6 if is_gouging else 1.1)
    vendor_db.append({
        'engine_id': eid,
        'part_name': part,
        'vendor': vendor,
        'base_market_price': round(base_cost, 2),
        'vendor_quoted_price': round(quoted_price, 2),
        'price_deviation_pct': round((quoted_price / base_cost - 1) * 100, 1),
        'is_price_gouging': is_gouging,
        'lead_time_days': np.random.randint(3, 21),
        'vendor_reliability_score': round(np.random.uniform(0.6, 0.99), 2),
    })

df_vendors = pd.DataFrame(vendor_db)
df_vendors.to_csv("DataSets/Synthetic/vendor_database.csv", index=False)

# Financial impact projection per engine
print("  → Computing Financial Loss Projections...")
financial_impact = []
for eid in range(1, n_engines + 1):
    engine_data = df_nasa[df_nasa['EngineID'] == eid]
    max_cycle = engine_data['Max_Cycle'].iloc[0]
    latest_rul = engine_data['RUL'].min()
    vendor_info = df_vendors[df_vendors['engine_id'] == eid].iloc[0]
    
    # Downtime cost: $5000/hour, assume 48h downtime for unplanned failure
    unplanned_downtime_cost = 5000 * 48  # $240,000
    planned_downtime_cost = 5000 * 8     # $40,000
    
    financial_impact.append({
        'engine_id': eid,
        'max_life_cycles': max_cycle,
        'part_cost': vendor_info['vendor_quoted_price'],
        'unplanned_failure_cost': unplanned_downtime_cost + vendor_info['vendor_quoted_price'],
        'planned_maintenance_cost': planned_downtime_cost + vendor_info['base_market_price'],
        'savings_if_predicted': round(unplanned_downtime_cost - planned_downtime_cost + 
                                       vendor_info['vendor_quoted_price'] - vendor_info['base_market_price'], 2),
    })

df_financial_impact = pd.DataFrame(financial_impact)
df_financial_impact.to_csv("DataSets/Synthetic/financial_impact.csv", index=False)

total_savings = df_financial_impact['savings_if_predicted'].sum()
print(f"    → Projected savings with predictive maintenance: ${total_savings:,.2f}")

print("  ✓ Phase 6 Complete.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "═" * 60)
print("  ENGINE COMPLETE — All artifacts saved to /models")
print("═" * 60)
print(f"""
  Models:       iforest, lof, xgb_fraud, xgb_rul, rf_rul
  SHAP:         3 explainers (RUL, Fraud XGB, IForest)
  Graph:        {len(cycles_found)} fraud rings, {G.number_of_nodes():,} nodes analyzed
  Vendors:      {len(df_vendors)} procurement records
  Financial:    ${total_savings:,.0f} projected savings
  
  Ready to launch: streamlit run app.py
""")
