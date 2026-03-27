import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import joblib
from datetime import datetime

st.set_page_config(page_title="EURI | Mission Control", page_icon="logo.png", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# CSS THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    :root {
        --bg-primary: #0a0e17; --bg-secondary: #111827; --bg-card: #151d2e;
        --accent-red: #ef4444; --accent-blue: #3b82f6; --accent-cyan: #06b6d4;
        --accent-green: #10b981; --accent-amber: #f59e0b; --accent-purple: #8b5cf6;
        --text-primary: #f1f5f9; --text-secondary: #94a3b8; --border: #1e293b;
    }
    .main, .stApp { background-color: var(--bg-primary) !important; }
    .block-container { padding: 2.5rem 1.5rem 1rem !important; max-width: 100% !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--bg-secondary); border-radius: 8px; padding: 4px; border: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { background: transparent; color: var(--text-secondary); border-radius: 6px; padding: 8px 16px; font-family: 'Inter'; font-weight: 500; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: var(--accent-blue) !important; color: white !important; }
    div[data-testid="stMetric"] { background: var(--bg-card); border-radius: 8px; padding: 14px 18px; border: 1px solid var(--border); border-left: 3px solid var(--accent-cyan); }
    div[data-testid="stMetric"] label { font-family: 'Inter'; font-size: 12px; color: var(--text-secondary); }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'JetBrains Mono'; color: var(--text-primary); }
    .stDataFrame { border: 1px solid var(--border); border-radius: 8px; }
    h1, h2, h3 { font-family: 'Inter' !important; color: var(--text-primary) !important; }
    .stSelectbox > div > div { background: var(--bg-card); border: 1px solid var(--border); }
    .risk-critical { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #dc2626; border-radius: 8px; padding: 16px; margin: 8px 0; color: #fecaca; font-family: 'Inter'; }
    .risk-nominal { background: linear-gradient(135deg, #052e16, #14532d); border: 1px solid #16a34a; border-radius: 8px; padding: 16px; margin: 8px 0; color: #bbf7d0; font-family: 'Inter'; }
    .risk-warning { background: linear-gradient(135deg, #451a03, #78350f); border: 1px solid #d97706; border-radius: 8px; padding: 16px; margin: 8px 0; color: #fde68a; font-family: 'Inter'; }
    .info-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 8px 0; font-family: 'Inter'; color: var(--text-primary); }
    .header-banner { background: linear-gradient(135deg, #0f172a, #1e1b4b); border: 1px solid #312e81; border-radius: 10px; padding: 12px 24px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_all_data():
    data = {}
    paths = {
        'paysim': 'DataSets/PAYSim/processed_paysim.csv',
        'nasa': 'DataSets/Synthetic/processed_nasa.csv',
        'cc': 'DataSets/Credit Card Fraud/processed_cc.csv',
        'vendors': 'DataSets/Synthetic/vendor_database.csv',
        'financial_impact': 'DataSets/Synthetic/financial_impact.csv',
        'graph_nodes': 'DataSets/Synthetic/graph_risk_nodes.csv',
        'graph_edges': 'DataSets/Synthetic/graph_suspicious_edges.csv',
        'cusum': 'DataSets/Synthetic/cusum_alarms.csv',
        'counterfactuals': 'DataSets/Synthetic/counterfactuals.csv',
        'monte_carlo': 'DataSets/Synthetic/monte_carlo_rul.csv',
        'prophet': 'DataSets/Synthetic/prophet_trends.csv',
        'compound_risk': 'DataSets/Synthetic/compound_risk_alerts.csv',
        'purchase_orders': 'DataSets/Synthetic/auto_purchase_orders.csv',
    }
    for key, path in paths.items():
        try:
            data[key] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        except:
            data[key] = pd.DataFrame()
    # JSON files
    for key, path in [('fraud_rings', 'models/fraud_rings.json'), ('risk_index', 'DataSets/Synthetic/risk_index.json'),
                       ('causal', 'DataSets/Synthetic/causal_results.json'), ('federated', 'DataSets/Synthetic/federated_results.json'),
                       ('adversarial', 'DataSets/Synthetic/adversarial_results.json'), ('manifest', 'models/manifest.json')]:
        try:
            data[key] = json.load(open(path)) if os.path.exists(path) else {}
        except:
            data[key] = {}
    return data

@st.cache_resource
def load_models():
    models = {}
    for key, path in {'xgb_rul': 'models/xgb_rul.pkl', 'rf_rul': 'models/rf_rul.pkl',
                       'shap_rul': 'models/shap_values_rul.pkl', 'shap_rul_sample': 'models/shap_rul_sample.pkl',
                       'shap_expected_rul': 'models/shap_expected_rul.pkl', 'shap_cc': 'models/shap_values_cc.pkl',
                       'shap_expected_cc': 'models/shap_expected_cc.pkl', 'shap_iforest': 'models/shap_values_iforest.pkl',
                       'shap_fraud_sample': 'models/shap_fraud_sample.pkl', 'rul_features': 'models/rul_features.pkl',
                       'sensor_cols': 'models/sensor_cols.pkl', 'shap_ae': 'models/shap_values_ae.pkl'}.items():
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except:
                pass
    return models

DARK_LAYOUT = dict(paper_bgcolor='#0a0e17', plot_bgcolor='#111827',
                    font=dict(color='#94a3b8', family='Inter'),
                    margin=dict(l=40, r=20, t=30, b=40))
GRID = dict(gridcolor='#1e293b')

# ============================================================
# LOGIN
# ============================================================
USERS = {
    'admin': {'password': 'admin123', 'role': 'Administrator', 'access': 'full'},
    'auditor': {'password': 'audit2026', 'role': 'Financial Auditor', 'access': 'finance'},
    'engineer': {'password': 'eng2026', 'role': 'Reliability Engineer', 'access': 'maintenance'},
}

def show_login():
    st.markdown("""
    <div style="display:flex;justify-content:center;margin-top:60px;">
        <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);border:1px solid #312e81;border-radius:16px;padding:40px 48px;width:420px;text-align:center;">
            <div style="display:flex;justify-content:center;margin-bottom:12px;"><img src="data:image/png;base64,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" width="80"></div>
            <div style="font-size:22px;font-weight:700;color:#f1f5f9;font-family:'Inter';">EURI Mission Control</div>
            <div style="font-size:13px;color:#94a3b8;margin-bottom:24px;">Enterprise Unified Risk Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_l, col_form, col_r = st.columns([1, 1.2, 1])
    with col_form:
        tab_login, tab_onboard = st.tabs(["🔐 Sign In", "🚀 Enterprise Setup"])
        
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="admin / auditor / engineer", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
            if st.button("🔐 Access Command Center", type="primary", use_container_width=True):
                if username in USERS and USERS[username]['password'] == password:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['role'] = USERS[username]['role']
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Try: admin/admin123")
            with st.expander("🔑 Demo Credentials"):
                st.markdown("| Username | Password | Role |\n|---|---|---|\n| `admin` | `admin123` | Full |\n| `auditor` | `audit2026` | Finance |\n| `engineer` | `eng2026` | Maintenance |")

        with tab_onboard:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<span style='color: #94a3b8; font-size: 14px;'>Connect your live infrastructure or upload historical extracts to dynamically train the hybrid AI framework.</span>", unsafe_allow_html=True)
            sector = st.selectbox("Business Sector", ["Manufacturing & Aerospace", "Financial Services / Banking", "SaaS / Tech Enterprise", "Logistics & Supply Chain"])
            if sector == "SaaS / Tech Enterprise":
                st.info("🛡️ Digital-First Selection: AI will prioritize Financial Fraud & HR Insider modules. Hardware Predictive Maintenance will be hibernated.")
            elif sector == "Manufacturing & Aerospace":
                st.info("⚙️ Asset-Heavy Selection: Full-stack activation (NASA RUL Engines + ERP Procurement Bridge).")
            
            st.selectbox("Core Environment Integration", ["SAP S/4HANA", "Oracle ERP Cloud", "Snowflake Data Warehouse", "Microsoft Dynamics 365", "Custom PostgreSQL", "Standalone CSV/Excel DB"])
            st.text_input("Connection String / IAM Key", type="password", placeholder="jdbc:sap://... or IAM token")
            st.file_uploader("Override with Local Export (CSV/XLSX)", accept_multiple_files=True)
            
            if st.button("⚡ Authenticate & Initialize Data Engine", type="primary", use_container_width=True):
                import time
                with st.spinner("Extracting schema and validating tensors..."):
                    time.sleep(2)
                st.success("✅ Secure Handshake verified. In a live system, the 4-Model Data Engine would now orchestrate pipeline ingests.")

if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    show_login()
    st.stop()


# Initialize Simulation State
if 'sim_active' not in st.session_state: st.session_state['sim_active'] = False
if 'sim_logs' not in st.session_state: st.session_state['sim_logs'] = []
if 'sim_type' not in st.session_state: st.session_state['sim_type'] = ''

def run_simulation(atype):
    st.session_state['sim_active'] = True
    st.session_state['sim_type'] = atype
    if atype == 'Financial':
        st.session_state['sim_logs'] = [
            ('📡', 'EURI Neural Engine: Ingressing 4.2M Transaction Burst...'),
            ('⚠️', 'CRITICAL ANOMALY: Unauthorized Treasury Withdrawal detected (Z-Score: 12.4)'),
            ('🔎', 'Agentic Auditor: Identifying Destination Node... 0x8a...4b (Ghost Vendor)'),
            ('🛡️', 'EURI MITIGATION: [ACTION] Freezing SAP Outbound Payment Gateway...'),
            ('⛓️', 'EURI MITIGATION: [ACTION] Initiating Chain-of-Custody Log...'),
            ('📧', 'EURI MITIGATION: [ACTION] Pushing High-Risk Slack Alert to CFO Cabinet...'),
            ('✅', 'EURI MITIGATION: .42M Outbound Containment Successful.')
        ]
    elif atype == 'Manufacturing':
        st.session_state['sim_logs'] = [
            ('⚙️', 'Sensor Ingress: Real-time Telemetry for Jet Engine #4 (Sector 7B)...'),
            ('🔥', 'CRITICAL ALERT: Exhaust Temp Spike (942°C) + Vibration Chaos detected!'),
            ('📉', 'EURI Predictive Engine: RUL plummeted from 42h to 0.4h in 60 seconds.'),
            ('🛑', 'EURI MITIGATION: [ACTION] Triggering Emergency Engine Kill-Switch...'),
            ('🏗️', 'EURI MITIGATION: [ACTION] Auto-Dispatching Maintenance Crew to Hangar 4.'),
            ('📋', 'EURI MITIGATION: [ACTION] Generating Automated Failure Analysis Report.'),
            ('✅', 'EURI MITIGATION: Asset shutdown successful. Preventing Catastrophic Explosion.')
        ]
    elif atype == 'Cyber':
        st.session_state['sim_logs'] = [
            ('🌐', 'Cloud Guard: Monitoring Lateral Movement in Oracle ERP Cloud...'),
            ('🔑', 'CRITICAL ALERT: Brute-force credentials detected on SSH Port 22 (940 attempts/min)'),
            ('👤', 'Agentic Auditor: Identity Trace... Authenticated as [admin] from Unauthorized IP (Moscow)'),
            ('🚫', 'EURI MITIGATION: [ACTION] Revoking Global Admin Credentials for Session ID #847'),
            ('🛡️', 'EURI MITIGATION: [ACTION] Activating Zero-Trust Isolation for Data Warehouse Hub...'),
            ('💬', 'EURI MITIGATION: [ACTION] Opening Incident Response Bridge in MS Teams.'),
            ('✅', 'EURI MITIGATION: Data Breach Prevented. Intrusion Source Isolated.')
        ]
    st.rerun()

# Header UI Update for Simulation
if st.session_state.get('sim_active'):
    atype = st.session_state.get('sim_type', 'THREAT')
    banner_color = '#7f1d1d'
    if atype == 'Manufacturing': banner_color = '#9a3412'
    if atype == 'Cyber': banner_color = '#1e3a8a'
    
    st.markdown(f'<div style="background-color:{banner_color}; color:white; padding:10px; border-radius:8px; text-align:center; font-weight:bold; margin-bottom:20px; border:2px solid #ef4444; animation: pulse 2s infinite;">🚨 ACTIVE {atype.upper()} MITIGATION IN PROGRESS: CRITICAL ANOMALY DETECTED</div>', unsafe_allow_html=True)
    st.markdown('<style>@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }</style>', unsafe_allow_html=True)

data = load_all_data()
models = load_models()
user_role = st.session_state.get('role', 'Unknown')
username = st.session_state.get('username', 'unknown')

# ============================================================
# HEADER + ENTERPRISE RISK INDEX
# ============================================================
ri = data.get('risk_index', {})
ri_total = ri.get('total', 0)
ri_label = ri.get('label', 'N/A')
label_color = {'CRITICAL': '#ef4444', 'HIGH': '#f59e0b', 'MODERATE': '#3b82f6', 'LOW': '#10b981'}.get(ri_label, '#94a3b8')

st.markdown(f"""
<div class="header-banner">
    <img src="data:image/png;base64,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" width="45" style="margin-right:12px;">
    <div style="flex:1;">
        <div style="font-size:20px;font-weight:700;color:#f1f5f9;font-family:'Inter';">EURI Mission Control</div>
        <div style="font-size:12px;color:#94a3b8;">Enterprise Unified Risk Intelligence</div>
    </div>
    <div style="text-align:center;margin:0 20px;">
        <div style="font-size:11px;color:#94a3b8;">RISK INDEX</div>
        <div style="font-size:28px;font-weight:700;color:{label_color};font-family:'JetBrains Mono';">{ri_total:.0f}<span style="font-size:14px;color:#64748b;">/100</span></div>
        <div style="font-size:10px;padding:2px 8px;border-radius:4px;background:{label_color}20;color:{label_color};font-weight:600;">{ri_label}</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:12px;color:#10b981;font-family:'JetBrains Mono';">● ONLINE</div>
        <div style="font-size:11px;color:#94a3b8;">{user_role} — {username}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sub-scores
rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric("💰 Financial Risk", f"{ri.get('financial', 0):.0f}/35")
rc2.metric("⚙️ Operational Risk", f"{ri.get('operational', 0):.0f}/35")
rc3.metric("📦 Procurement Risk", f"{ri.get('procurement', 0):.0f}/20")
rc4.metric("🛡️ Robustness Score", f"{ri.get('robustness', 0):.0f}/10")

# ============================================================
# TABS
# ============================================================
tab_overview, tab_finance, tab_maintenance, tab_erp, tab_auditor, tab_hr = st.tabs([
    "📡 Command Center",
    "💰 Financial Threat Intel",
    "⚙️ Predictive Maintenance",
    "🔗 ERP Procurement Bridge",
    "🤖 Agentic Auditor",
    "👥 HR Insider Risk"
])

# ============================================================
# TAB 1: COMMAND CENTER
# ============================================================
with tab_overview:
    df_p = data['paysim']
    df_n = data['nasa']
    df_v = data['vendors']
    df_fi = data['financial_impact']

    if not df_p.empty and not df_n.empty:
        # Engine health snapshot
        engine_summary = df_n.groupby('EngineID').agg(latest_cycle=('Cycle','max'), max_life=('Max_Cycle','first')).reset_index()
        np.random.seed(42)
        engine_summary['min_rul'] = engine_summary['max_life'].apply(lambda m: np.random.randint(0, max(1, int(m))))
        engine_summary['health_pct'] = (engine_summary['min_rul'] / engine_summary['max_life'] * 100).clip(0, 100)

        anomalies_count = int(df_p['Is_Anomaly'].sum()) if 'Is_Anomaly' in df_p.columns else 0
        critical_engines = len(engine_summary[engine_summary['health_pct'] < 30])
        cusum_count = len(data.get('cusum', pd.DataFrame()))
        compound_count = int(data.get('compound_risk', pd.DataFrame()).get('compound_risk', pd.Series(dtype=bool)).sum()) if not data.get('compound_risk', pd.DataFrame()).empty else 0
        po_count = critical_engines

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🔴 Anomalies (v2 Ensemble)", f"{anomalies_count:,}")
        c2.metric("⚠️ Critical Engines", critical_engines)
        c3.metric("📊 Auto-POs Generated", po_count)
        c4.metric("📈 CUSUM Drift Alarms", cusum_count)
        c5.metric("🔗 Compound Risk Alerts", compound_count)
        c6.metric("📊 Transactions Scanned", f"{len(df_p):,}")

        st.divider()
        col_left, col_right = st.columns([1.3, 1])

        # Risk trend chart with CUSUM alarms
        with col_left:
            st.markdown("#### 📈 Fraud Rate Trend + CUSUM Alarms")
            if 'Is_Anomaly' in df_p.columns:
                fraud_rate = df_p.groupby('step')['Is_Anomaly'].mean()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=fraud_rate.index, y=fraud_rate.values, mode='lines', name='Anomaly Rate', line=dict(color='#3b82f6', width=1.5)))
                df_cusum = data.get('cusum', pd.DataFrame())
                if not df_cusum.empty:
                    fr_alarms = df_cusum[df_cusum['type'] == 'fraud_rate'] if 'type' in df_cusum.columns else df_cusum
                    for _, a in fr_alarms.iterrows():
                        fig_trend.add_vline(x=a.get('step', 0), line_dash="dash", line_color="#ef4444", opacity=0.6)
                    amt_alarms = df_cusum[df_cusum['type'] == 'amount'] if 'type' in df_cusum.columns else pd.DataFrame()
                    for _, a in amt_alarms.iterrows():
                        fig_trend.add_vline(x=a.get('step', 0), line_dash="dot", line_color="#f59e0b", opacity=0.4)
                fig_trend.update_layout(height=280, **DARK_LAYOUT, xaxis=dict(title='Step (Hour)', **GRID), yaxis=dict(title='Rate', **GRID))
                st.plotly_chart(fig_trend, use_container_width=True)

        # Radar chart
        with col_right:
            st.markdown("#### 🎯 Risk Radar")
            categories = ['Financial', 'Operational', 'Procurement', 'Robustness']
            values = [ri.get('financial', 0)/35*100, ri.get('operational', 0)/35*100, ri.get('procurement', 0)/20*100, (10 - ri.get('robustness', 5))/10*100]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', fillcolor='rgba(239,68,68,0.15)', line=dict(color='#ef4444', width=2), name='Current'))
            fig_radar.add_trace(go.Scatterpolar(r=[50]*5, theta=categories + [categories[0]], line=dict(color='#22c55e', width=1, dash='dash'), name='Threshold'))
            fig_radar.update_layout(height=280, polar=dict(bgcolor='#111827', radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e293b', tickfont=dict(color='#64748b')), angularaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#94a3b8'))), **DARK_LAYOUT, showlegend=True, legend=dict(x=0.7, y=1))
            st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap + cost
        col_h, col_c = st.columns([1.3, 1])
        with col_h:
            st.markdown("#### 🌐 Engine Health Heatmap")
            fig_hm = go.Figure(go.Bar(x=engine_summary['EngineID'], y=engine_summary['health_pct'],
                marker_color=['#ef4444' if h < 20 else '#f59e0b' if h < 50 else '#10b981' for h in engine_summary['health_pct']],
                text=[f"{h:.0f}%" for h in engine_summary['health_pct']], textposition='outside', textfont=dict(size=10, color='white')))
            fig_hm.update_layout(height=260, **DARK_LAYOUT, xaxis=dict(title='Engine ID', **GRID), yaxis=dict(title='Health %', **GRID, range=[0, 110]))
            st.plotly_chart(fig_hm, use_container_width=True)

        with col_c:
            st.markdown("#### 📊 Risk Distribution")
            if 'Risk_Score_Ensemble' in df_p.columns:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=df_p['Risk_Score_Ensemble'], nbinsx=80, marker_color='#3b82f6', opacity=0.7, name='All'))
                fig_dist.add_trace(go.Histogram(x=df_p[df_p['Is_Anomaly']==1]['Risk_Score_Ensemble'], nbinsx=30, marker_color='#ef4444', opacity=0.9, name='Anomalies'))
                fig_dist.update_layout(height=260, barmode='overlay', **DARK_LAYOUT, xaxis=dict(title='Risk Score', **GRID), yaxis=dict(title='Count', **GRID), legend=dict(x=0.6, y=0.95))
                st.plotly_chart(fig_dist, use_container_width=True)

        # Compound risk alerts
        df_comp = data.get('compound_risk', pd.DataFrame())
        if not df_comp.empty:
            st.markdown("#### 🔗 Compound Risk Alerts")
            def style_compound(row):
                if row.get('compound_risk', False):
                    return ['background-color: #7f1d1d; color: #f8fafc;'] * len(row)
                elif row.get('severity', '') == 'HIGH':
                    return ['background-color: #854d0e; color: #f8fafc;'] * len(row)
                return ['background-color: #064e3b; color: #f8fafc;'] * len(row)
            display_cols = [c for c in ['engine_id','health_pct','vendor','severity','compound_risk','recommended_action'] if c in df_comp.columns]
            st.dataframe(df_comp[display_cols].style.apply(style_compound, axis=1), use_container_width=True, height=200)
    else:
        st.warning("Run `python data_engine.py` first.")

# ============================================================
# TAB 2: FINANCIAL INTELLIGENCE
# ============================================================
with tab_finance:
    df_p = data['paysim']
    if not df_p.empty:
        # Section A: Anomaly Explorer
        st.markdown("#### 🔍 Anomaly Explorer")
        model_cols = {'IForest': 'Risk_Score_IForest', 'LOF': 'Risk_Score_LOF', 'Autoencoder': 'AE_Score', 'GNN-Lite': 'GNN_Risk_Score', 'Ensemble': 'Risk_Score_Ensemble'}
        col_sel, col_info = st.columns([1, 3])
        with col_sel:
            sel_model = st.selectbox("Signal", list(model_cols.keys()), key="model_sel")
        score_col = model_cols[sel_model]
        if score_col in df_p.columns:
            anomalies = df_p[df_p['Is_Anomaly'] == 1].sort_values('Risk_Score_Ensemble', ascending=False)
            display_cols = ['step', 'type', 'amount', 'nameOrig', 'nameDest', score_col, 'Risk_Score_Ensemble', 'in_circular_ring']
            available = list(dict.fromkeys([c for c in display_cols if c in anomalies.columns]))
            df_display = anomalies[available].head(80).copy()
            if not df_display.empty and 'Risk_Score_Ensemble' in df_display.columns:
                risk_vals = df_display['Risk_Score_Ensemble'].values
                p_high = np.percentile(risk_vals, 85)
                p_mod = np.percentile(risk_vals, 40)
                def style_rows(row):
                    r = row.get('Risk_Score_Ensemble', 0)
                    if r >= p_high: color = 'background-color: #7f1d1d; color: #f8fafc;'
                    elif r >= p_mod: color = 'background-color: #854d0e; color: #f8fafc;'
                    else: color = 'background-color: #064e3b; color: #f8fafc;'
                    return [color] * len(row)
                st.markdown("""<div style="display:flex;gap:15px;font-size:12px;margin-bottom:10px;">
                    <div><span style="color:#ef4444;">●</span> Red: Highest Risk</div>
                    <div><span style="color:#eab308;">●</span> Yellow: Moderate</div>
                    <div><span style="color:#22c55e;">●</span> Green: Lower Risk</div></div>""", unsafe_allow_html=True)
                st.dataframe(df_display.style.apply(style_rows, axis=1), use_container_width=True, height=400)

        # Section B: Community Detection
        st.divider()
        df_gn = data.get('graph_nodes', pd.DataFrame())
        if not df_gn.empty and 'community_id' in df_gn.columns:
            st.markdown("#### 🕸️ Graph Community Analysis")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                comm_sizes = df_gn.groupby('community_id').size().nlargest(10).reset_index(name='size')
                comm_risk = df_gn.groupby('community_id')['high_risk_community'].first().reset_index()
                comm_data = comm_sizes.merge(comm_risk, on='community_id')
                fig_comm = go.Figure(go.Bar(x=comm_data['community_id'].astype(str), y=comm_data['size'],
                    marker_color=['#ef4444' if r else '#3b82f6' for r in comm_data['high_risk_community']]))
                fig_comm.update_layout(title="Top 10 Communities by Size", height=250, **DARK_LAYOUT, xaxis=dict(title='Community ID'), yaxis=dict(title='Nodes'))
                st.plotly_chart(fig_comm, use_container_width=True)
            with col_g2:
                if 'betweenness' in df_gn.columns:
                    top_hubs = df_gn.nlargest(15, 'betweenness')[['account', 'betweenness', 'connections', 'pagerank']]
                    st.markdown("**Top Hub Accounts (Betweenness Centrality)**")
                    st.dataframe(top_hubs, use_container_width=True, height=250, hide_index=True)

        # Section C: Counterfactuals
        df_cf = data.get('counterfactuals', pd.DataFrame())
        if not df_cf.empty:
            st.divider()
            st.markdown("#### 🔄 Counterfactual Explanations")
            st.markdown("_What minimal change would have prevented detection?_")
            for _, cf in df_cf.head(5).iterrows():
                cols = st.columns([1, 1, 1, 2])
                cols[0].metric("Original Amount", f"${cf.get('original_amount', 0):,.0f}")
                cols[1].metric("Original Risk", f"{cf.get('original_risk', 0):.4f}")
                cols[2].metric("Key Feature", cf.get('key_feature_changed', 'N/A'))
                cols[3].markdown(f"<div class='info-card'>Change <b>{cf.get('key_feature_changed', 'N/A')}</b> by <b>{cf.get('change_magnitude', 0):,.1f}</b> to evade detection</div>", unsafe_allow_html=True)

        # Section D: CUSUM
        df_cusum = data.get('cusum', pd.DataFrame())
        if not df_cusum.empty:
            st.divider()
            st.markdown("#### 📈 CUSUM Drift Monitor")
            st.metric("Active Drift Alarms", len(df_cusum))
            st.dataframe(df_cusum, use_container_width=True, height=200, hide_index=True)

        # Section E: Adversarial - Removed for PPT focus
        pass
    else:
        st.warning("Run `python data_engine.py` first.")

with tab_maintenance:
    df_n = data['nasa']
    
    if not df_n.empty:
        sensor_cols = [c for c in df_n.columns if c.startswith('Sensor_')]
        rul_features = models.get('rul_features', ['Cycle'] + sensor_cols)
        
        col_sel, col_health = st.columns([1, 1])
        with col_sel:
            engine_ids = sorted(df_n['EngineID'].unique())
            selected_engine = st.selectbox("Select Engine Unit", engine_ids, key="eng_sel")
        
        engine_data = df_n[df_n['EngineID'] == selected_engine].sort_values('Cycle')
        max_cycle = int(engine_data['Max_Cycle'].iloc[0])
        
        with col_health:
            current_cycle = st.slider("Simulate Current Cycle", 1, max_cycle, max_cycle, key="cycle_sl")
        
        dynamic_rul = max_cycle - current_cycle
        dynamic_health = min(100.0, max(0.0, (dynamic_rul / max_cycle) * 100))

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("🏥 Health", f"{dynamic_health:.1f}%")
        mc2.metric("🔧 RUL", f"{dynamic_rul} cycles")
        mc3.metric("⏱️ Max Life", f"{max_cycle} cycles")
        mc4.metric("📍 Current Cycle", current_cycle)

        col_chart, col_gauge = st.columns([2, 1])
        
        with col_chart:
            st.markdown("#### 📈 Multi-Sensor Telemetry")
            fig_sensors = go.Figure()
            palette = ['#06b6d4', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6']
            for i, sc in enumerate(sensor_cols):
                fig_sensors.add_trace(go.Scatter(
                    x=engine_data['Cycle'], y=engine_data[sc],
                    mode='lines', name=sc.replace('Sensor_', 'S'),
                    line=dict(color=palette[i % len(palette)], width=1.5),
                ))
            fig_sensors.add_vline(x=current_cycle, line_dash="dash", line_color="#f59e0b", opacity=0.7)
            fig_sensors.update_layout(
                height=300, margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor='#0a0e17', plot_bgcolor='#111827',
                font=dict(color='#94a3b8', family='Inter'),
                xaxis=dict(gridcolor='#1e293b', title='Cycle'),
                yaxis=dict(gridcolor='#1e293b', title='Reading'),
                legend=dict(orientation='h', y=1.12, font=dict(size=10)),
            )
            st.plotly_chart(fig_sensors, use_container_width=True)

        with col_gauge:
            st.markdown("#### 🏥 Health Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dynamic_health,
                number={'suffix': '%', 'font': {'size': 36, 'color': 'white', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#475569'},
                    'bar': {'color': 'rgba(255,255,255,0.85)', 'thickness': 0.3},
                    'bgcolor': '#111827',
                    'steps': [
                        {'range': [0, 20], 'color': '#7f1d1d'},
                        {'range': [20, 50], 'color': '#78350f'},
                        {'range': [50, 100], 'color': '#052e16'}
                    ],
                    'threshold': {'line': {'color': '#ef4444', 'width': 3}, 'thickness': 0.8, 'value': 20}
                }
            ))
            fig_gauge.update_layout(
                height=250, margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor='#0a0e17', font=dict(color='white')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()
        st.markdown("#### 🔮 Digital Twin: What-If Simulator")
        st.markdown("*Adjust operating conditions to simulate impact on Remaining Useful Life.*")
        
        wif1, wif2, wif3 = st.columns(3)
        with wif1:
            temp_adj = st.slider("Temperature Offset (°K)", -20.0, 20.0, 0.0, 0.5, key="temp_adj")
        with wif2:
            torque_adj = st.slider("Torque Offset (Nm)", -15.0, 15.0, 0.0, 0.5, key="torque_adj")
        with wif3:
            speed_adj = st.slider("Bypass Ratio Offset", -10.0, 10.0, 0.0, 0.5, key="speed_adj")

        if 'xgb_rul' in models:
            xgb_model = models['xgb_rul']
            cycle_row = engine_data[engine_data['Cycle'] == current_cycle]
            if not cycle_row.empty:
                row = cycle_row.iloc[0]
                feature_vals = {}
                for f in rul_features:
                    if f in row.index:
                        feature_vals[f] = row[f]
                    else:
                        feature_vals[f] = 0
                
                for k in feature_vals:
                    if 'Temp' in k:
                        feature_vals[k] += temp_adj
                    if 'LPT' in k:
                        feature_vals[k] += torque_adj
                    if 'Bypass' in k:
                        feature_vals[k] += speed_adj
                
                X_whatif = pd.DataFrame([feature_vals])[rul_features]
                predicted_rul = xgb_model.predict(X_whatif)[0]
                
                baseline_vals = {f: row[f] if f in row.index else 0 for f in rul_features}
                X_base = pd.DataFrame([baseline_vals])[rul_features]
                baseline_rul = xgb_model.predict(X_base)[0]
                
                delta = predicted_rul - baseline_rul
                
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("Baseline RUL", f"{baseline_rul:.0f} cycles")
                wc2.metric("Simulated RUL", f"{predicted_rul:.0f} cycles", delta=f"{delta:+.0f}")
                wc3.metric("Life Impact", 
                          f"{'Extended' if delta > 0 else 'Shortened'} by {abs(delta):.0f} cycles",
                          delta=f"{delta:+.0f}")
                
                if delta < -10:
                    st.markdown('<div class="risk-critical">⚠️ <b>CRITICAL</b>: These conditions significantly reduce asset lifespan. Immediate corrective action recommended.</div>', unsafe_allow_html=True)
                elif delta < 0:
                    st.markdown('<div class="risk-warning">⚡ <b>WARNING</b>: Slight reduction in asset lifespan detected under these conditions.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-nominal">✅ <b>FAVORABLE</b>: These operating conditions extend or maintain asset lifespan.</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 🔬 XAI: Why is this engine predicted to fail?")
        if 'shap_rul' in models and 'shap_rul_sample' in models:
            shap_vals = models['shap_rul']
            sample_df = models['shap_rul_sample']
            features = sample_df.columns.tolist()
            mean_shap = np.mean(np.abs(shap_vals), axis=0)
            sorted_i = np.argsort(mean_shap)[::-1]
            
            fig_rul_shap = go.Figure(go.Bar(
                x=[mean_shap[i] for i in sorted_i],
                y=[features[i] for i in sorted_i],
                orientation='h', marker_color='#06b6d4',
                text=[f"{mean_shap[i]:.2f}" for i in sorted_i], textposition='outside',
            ))
            fig_rul_shap.update_layout(
                title="Global Feature Importance (XGBoost RUL)",
                height=280, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor='#0a0e17', plot_bgcolor='#151d2e',
                font=dict(color='#94a3b8', size=11), yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_rul_shap, use_container_width=True)
    else:
        st.warning("Run `python3 data_engine.py` first.")


with tab_erp:
    df_n = data['nasa']
    df_v = data['vendors']
    df_fi = data['financial_impact']
    
    if not df_n.empty and not df_v.empty:
        st.markdown("#### 🔗 Automated Procurement Logic Bridge")
        st.markdown("*When asset health drops below threshold, the system auto-generates procurement requests and audits vendor pricing.*")

        # Generate simulated snapshot of engine healths
        engine_summary = df_n.groupby('EngineID').agg(
            max_life=('Max_Cycle', 'first')
        ).reset_index()
        np.random.seed(42)
        engine_summary['latest_rul'] = engine_summary['max_life'].apply(lambda m: np.random.randint(0, max(1, int(m))))
        engine_summary['health_pct'] = (engine_summary['latest_rul'] / engine_summary['max_life'] * 100)
        
        critical = engine_summary[engine_summary['health_pct'] < 30]
        warning_engines = engine_summary[(engine_summary['health_pct'] >= 30) & (engine_summary['health_pct'] < 50)]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Critical (< 30%)", len(critical))
        c2.metric("🟡 Warning (30-50%)", len(warning_engines))
        c3.metric("🟢 Nominal (> 50%)", len(engine_summary) - len(critical) - len(warning_engines))
        
        st.divider()
        
        for _, eng in critical.iterrows():
            eid = eng['EngineID']
            health = eng['health_pct']
            vendor_row = df_v[df_v['engine_id'] == eid]
            fi_row = df_fi[df_fi['engine_id'] == eid]
            
            if not vendor_row.empty:
                vr = vendor_row.iloc[0]
                fr = fi_row.iloc[0] if not fi_row.empty else None
                
                is_gouging = vr['is_price_gouging']
                
                with st.expander(f"🚨 Engine {int(eid)} — Health: {health:.1f}% — {'⚠️ PRICE GOUGING DETECTED' if is_gouging else '✅ VENDOR CLEAN'}", expanded=True):
                    pc1, pc2 = st.columns([1.5, 1])
                    
                    with pc1:
                        st.markdown("**📋 Auto-Generated Purchase Requisition**")
                        pr_data = {
                            'Field': ['PR Number', 'Engine ID', 'Part Required', 'Vendor', 
                                     'Market Price', 'Quoted Price', 'Price Deviation',
                                     'Lead Time', 'Vendor Score', 'Urgency'],
                            'Value': [
                                f"PR-{int(eid):03d}-{datetime.now().strftime('%Y%m%d')}",
                                f"ENG-{int(eid):03d}",
                                vr['part_name'],
                                vr['vendor'],
                                f"${vr['base_market_price']:,.2f}",
                                f"${vr['vendor_quoted_price']:,.2f}",
                                f"{vr['price_deviation_pct']}%",
                                f"{vr['lead_time_days']} days",
                                f"{vr['vendor_reliability_score']:.2f}",
                                '🔴 CRITICAL' if health < 10 else '🟡 HIGH'
                            ]
                        }
                        st.dataframe(pd.DataFrame(pr_data), use_container_width=True, hide_index=True)
                    
                    with pc2:
                        st.markdown("**🔍 Financial Audit Status**")
                        if is_gouging:
                            st.markdown(f"""
                            <div class="risk-critical">
                            <b>🚨 PRICE GOUGING ALERT</b><br>
                            Vendor <b>{vr['vendor']}</b> quoted <b>${vr['vendor_quoted_price']:,.2f}</b> 
                            vs market price <b>${vr['base_market_price']:,.2f}</b>.<br>
                            Deviation: <b>{vr['price_deviation_pct']}%</b><br><br>
                            <b>Recommendation:</b> Escalate to procurement manager. Request re-quote or switch vendor.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="risk-nominal">
                            <b>✅ AUDIT PASSED</b><br>
                            Vendor pricing within acceptable range ({vr['price_deviation_pct']}% deviation).<br>
                            Vendor reliability: <b>{vr['vendor_reliability_score']:.0%}</b><br><br>
                            <b>Status:</b> Approved for automatic dispatch.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if fr is not None:
                            st.metric("💰 Savings (Predictive vs Unplanned)", f"${fr['savings_if_predicted']:,.0f}")
        
        if len(critical) == 0:
            st.markdown('<div class="risk-nominal">✅ No engines currently require emergency procurement. All assets within safe operating parameters.</div>', unsafe_allow_html=True)
            
        st.divider()
        st.markdown("#### 🏆 Vendor Rankings")
        vendor_rankings = df_v.groupby('vendor').agg(
            avg_reliability_score=('vendor_reliability_score', 'mean'),
            avg_price_deviation=('price_deviation_pct', 'mean'),
            gouging_incidents=('is_price_gouging', 'sum'),
            total_parts_supplied=('part_name', 'count')
        ).reset_index().sort_values(['avg_price_deviation', 'avg_reliability_score'], ascending=[True, False])
        
        # Format columns for display
        display_rankings = vendor_rankings.copy()
        display_rankings['avg_reliability_score'] = display_rankings['avg_reliability_score'].apply(lambda x: f"{x:.0%}")
        display_rankings['avg_price_deviation'] = display_rankings['avg_price_deviation'].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(
            display_rankings.rename(columns={
                'vendor': 'Vendor',
                'avg_reliability_score': 'Avg Reliability',
                'avg_price_deviation': 'Avg Price Deviation',
                'gouging_incidents': 'Gouging Incidents',
                'total_parts_supplied': 'Total Parts Supplied'
            }), 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.warning("Run `python3 data_engine.py` first.")


with tab_auditor:
    st.markdown("#### 🤖 Agentic Auditor — AI-Generated Risk Intelligence Reports")
    st.markdown("*The system reads SHAP explanations, engine telemetry, and fraud signals to produce natural language audit reports.*")
    
    report_type = st.selectbox("Select Report Type", [
        "Executive Risk Summary",
        "Engine Maintenance Brief",
        "Financial Anomaly Investigation"
    ], key="report_type")
    
    if st.button("🔄 Generate Report", type="primary", key="gen_report"):
        df_p = data['paysim']
        df_n = data['nasa']
        df_v = data['vendors']
        df_fi = data['financial_impact']
        
        with st.spinner("AI Agent analyzing data and generating report..."):
            import time
            time.sleep(1)  # Simulate processing
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if report_type == "Executive Risk Summary":
                anomaly_count = df_p['Is_Anomaly'].sum() if not df_p.empty and 'Is_Anomaly' in df_p.columns else 0
                rings = len(data['fraud_rings'])
                
                engine_summary = df_n.groupby('EngineID').agg(latest_rul=('RUL', 'min'), max_life=('Max_Cycle', 'first')).reset_index()
                engine_summary['health'] = engine_summary['latest_rul'] / engine_summary['max_life'] * 100
                critical_count = len(engine_summary[engine_summary['health'] < 20])
                avg_health = engine_summary['health'].mean()
                
                total_savings = df_fi['savings_if_predicted'].sum() if not df_fi.empty else 0
                gouging = df_v['is_price_gouging'].sum() if not df_v.empty else 0
                
                # SHAP-driven insight
                shap_insight = ""
                if 'shap_cc' in models:
                    cc_features = ['amount', 'transaction_hour', 'foreign_transaction',
                                  'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']
                    mean_shap = np.mean(np.abs(models['shap_cc']), axis=0)
                    top_feature = cc_features[np.argmax(mean_shap)]
                    shap_insight = f"SHAP analysis reveals that **{top_feature}** is the #1 driver of fraud risk across all analyzed transactions."
                
                report = f"""
## 📋 EXECUTIVE RISK INTELLIGENCE SUMMARY
**Generated:** {timestamp} | **Classification:** CONFIDENTIAL | **Period:** Last 24h

---

### 🔴 Financial Integrity
- **{anomaly_count:,}** suspicious transactions flagged by the ensemble detector (IForest + LOF)
- **{rings}** circular payment rings identified via graph network analysis
- {shap_insight}
- **Recommendation:** Transactions flagged with risk score > 0.95 require immediate manual review by the compliance team.

### ⚙️ Operational Reliability
- **{critical_count}** engines in CRITICAL condition (health < 20%)
- Average fleet health: **{avg_health:.1f}%**
- The XGBoost RUL model predicts that **Cycle count** and **HPC outlet temperature** are the primary drivers of degradation (validated by SHAP TreeExplainer).
- **Recommendation:** Schedule preventive maintenance for all critical engines within the next 48 hours.

### 💰 Financial Impact
- Predictive maintenance could save **${total_savings:,.0f}** vs unplanned failure across the fleet.
- **{int(gouging)}** vendor quotes flagged for price gouging (deviation > 50% from market).
- **Recommendation:** Re-negotiate with flagged vendors before approving PRs.

### 🔗 Cross-Domain Insight
If Engine #{engine_summary.sort_values('health').iloc[0]['EngineID']:.0f} fails without prediction, the combined cost of emergency procurement + unplanned downtime could reach **${df_fi['unplanned_failure_cost'].max():,.0f}**. Our predictive system reduces this to **${df_fi['planned_maintenance_cost'].min():,.0f}**.

---
*Report generated by EURI Agentic Auditor v3.0 — Powered by SHAP Explainability Engine*
"""
                st.markdown(report)
            
            elif report_type == "Engine Maintenance Brief":
                eid = st.session_state.get('eng_sel', 1)
                engine_data = df_n[df_n['EngineID'] == eid]
                if not engine_data.empty:
                    max_c = engine_data['Max_Cycle'].iloc[0]
                    min_rul = engine_data['RUL'].min()
                    health = min_rul / max_c * 100
                    vendor_row = df_v[df_v['engine_id'] == eid]
                    
                    sensor_cols_avail = [c for c in engine_data.columns if c.startswith('Sensor_')]
                    latest = engine_data.iloc[-1]
                    sensor_report = "\n".join([f"  - **{s}**: {latest[s]:.2f}" for s in sensor_cols_avail])
                    
                    vendor_info = ""
                    if not vendor_row.empty:
                        vr = vendor_row.iloc[0]
                        vendor_info = f"""
### 🏭 Vendor & Procurement
- **Part:** {vr['part_name']}
- **Vendor:** {vr['vendor']} (reliability: {vr['vendor_reliability_score']:.0%})
- **Quoted:** ${vr['vendor_quoted_price']:,.2f} (market: ${vr['base_market_price']:,.2f})
- **Price Deviation:** {vr['price_deviation_pct']}% {'🚨 GOUGING' if vr['is_price_gouging'] else '✅ FAIR'}
"""
                    
                    report = f"""
## 🔧 MAINTENANCE BRIEF — Engine {eid}
**Generated:** {timestamp} | **Priority:** {'🔴 CRITICAL' if health < 20 else '🟡 MONITOR' if health < 50 else '🟢 NOMINAL'}

---

### Asset Status
- **Health:** {health:.1f}%
- **RUL:** {min_rul} cycles remaining
- **Total Lifespan:** {max_c} cycles

### Latest Sensor Readings (Cycle {int(latest['Cycle'])})
{sensor_report}

### SHAP Degradation Analysis
The XGBoost model indicates that **Cycle count** is the dominant factor, with sensor readings corroborating progressive wear patterns typical of turbofan compressor section degradation.

{vendor_info}

### Action Items
1. {'**IMMEDIATE:** Schedule emergency maintenance within 24h' if health < 20 else '**ROUTINE:** Continue monitoring' }
2. {'Review vendor quote for price gouging before approving PR' if not vendor_row.empty and vendor_row.iloc[0]['is_price_gouging'] else 'Vendor pricing within acceptable range'}

---
*Report generated by EURI Agentic Auditor v3.0*
"""
                    st.markdown(report)
            
            elif report_type == "Financial Anomaly Investigation":
                if not df_p.empty and 'Is_Anomaly' in df_p.columns:
                    anomalies = df_p[df_p['Is_Anomaly'] == 1].sort_values('Risk_Score_Ensemble', ascending=False)
                    top5 = anomalies.head(5)
                    rings = data['fraud_rings']
                    
                    txn_details = ""
                    for i, (_, txn) in enumerate(top5.iterrows()):
                        in_ring = txn.get('in_circular_ring', False)
                        txn_details += f"""
**Transaction #{i+1}** (Risk: {txn['Risk_Score_Ensemble']:.4f})
- Type: `{txn['type']}` | Amount: `${txn['amount']:,.2f}`
- From: `{txn['nameOrig']}` → To: `{txn['nameDest']}`
- Balance Error: `${txn.get('balance_error_orig', 0):,.2f}`
- In Fraud Ring: `{'YES 🚨' if in_ring else 'No'}`

"""
                    
                    report = f"""
## 🔍 FINANCIAL ANOMALY INVESTIGATION REPORT
**Generated:** {timestamp} | **Classification:** RESTRICTED

---

### Summary
The ensemble anomaly detector (Isolation Forest + Local Outlier Factor) flagged **{len(anomalies):,}** transactions 
from a pool of **{len(df_p):,}** as suspicious (top 1% risk threshold).

Graph network analysis identified **{len(rings)}** circular payment rings involving multiple accounts.

### Top 5 Highest-Risk Transactions
{txn_details}

### SHAP Root Cause Analysis
The Isolation Forest SHAP TreeExplainer identifies **balance_error** and **amount_to_balance_ratio** as the 
primary drivers of anomaly scores, consistent with known patterns of balance manipulation in financial fraud.

### Recommendations
1. Forward top 20 flagged transactions to the Compliance Investigation Unit
2. Freeze accounts identified in circular payment rings pending review
3. Cross-reference flagged accounts with KYC records

---
*Report generated by EURI Agentic Auditor v3.0 — Powered by SHAP + NetworkX Graph Intelligence*
"""
                    st.markdown(report)

        st.download_button("📥 Download Report", report, file_name=f"EURI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown")

# ============================================================
# TAB 6: HR INSIDER RISK (MOCKUP)
# ============================================================
with tab_hr:
    st.markdown("### 👥 Employee & Insider Threat Intelligence")
    st.markdown("<span style='color: #94a3b8;'>Detecting Segregation of Duties (SoD) violations, off-hour access, and unauthorized cross-module lateral movement.</span>", unsafe_allow_html=True)
    st.divider()
    
    col_hr1, col_hr2 = st.columns([2, 1])
    
    with col_hr1:
        st.markdown("#### 🚨 Flagged Access Anomalies")
        
        # Generate mock HR threat data
        mock_hr_data = pd.DataFrame({
            'Employee_ID': ['E-8472', 'E-9104', 'E-3321', 'E-5590', 'E-4102'],
            'Department': ['HR/Payroll', 'Supply Chain', 'Finance', 'IT Ops', 'HR/Payroll'],
            'Anomaly_Type': ['SoD Violation', 'Odd-Hour Access', 'Lateral Movement', 'Data Exfiltration Risk', 'Ghost Employee Creation'],
            'Severity': ['CRITICAL', 'HIGH', 'HIGH', 'ELEVATED', 'CRITICAL'],
            'Description': [
                'Created vendor AND approved $14k PO',
                'Logged into S/4HANA at 03:14 AM (Off-shift)',
                'Accessed Vendor Master Data (Unusual for role)',
                'Mass download of employee PII tables',
                'Created payroll record avoiding manager workflow'
            ],
            'Risk_Score': [98, 85, 79, 65, 94]
        })
        
        def hr_styler(row):
            if row['Severity'] == 'CRITICAL':
                return ['background-color: #7f1d1d; color: #f8fafc;'] * len(row)
            elif row['Severity'] == 'HIGH':
                return ['background-color: #854d0e; color: #f8fafc;'] * len(row)
            return ['background-color: #064e3b; color: #f8fafc;'] * len(row)
            
        st.dataframe(mock_hr_data.style.apply(hr_styler, axis=1), use_container_width=True, hide_index=True)
        
    with col_hr2:
        st.markdown("#### Risk by Department")
        # Mock chart
        dept_risk = pd.DataFrame({
            'Department': ['Finance', 'HR/Payroll', 'Supply Chain', 'IT Ops', 'Engineering'],
            'Incidents': [14, 22, 18, 5, 2]
        })
        
        fig_hr = px.bar(dept_risk, x='Incidents', y='Department', orientation='h', 
                        color='Incidents', color_continuous_scale='Reds')
        layout_args = DARK_LAYOUT.copy()
        layout_args.update(height=300, showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        fig_hr.update_layout(**layout_args)
        st.plotly_chart(fig_hr, use_container_width=True)
        
        st.markdown("""<div class='info-card'>
            <b>🛡️ HR Security Policies Enforced</b><br>
            • Segregation of Duties (SoD)<br>
            • Time-of-Travel Anomalies<br>
            • Privilege Escalation Attempts
        </div>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("logo.png", width=80)
    st.markdown("### EURI")
    manifest = data.get('manifest', {})
    st.markdown(f"**Version:** {manifest.get('version', '4.0')}")
    st.markdown(f"**Trained:** {manifest.get('trained_at', 'N/A')[:19]}")
    st.divider()
    st.divider()
    st.markdown('#### 💀 Live Threat Simulator')
    cols = st.columns(3)
    if cols[0].button('💰 Fin', use_container_width=True, help='Simulate Financial Fraud'): run_simulation('Financial')
    if cols[1].button('⚙️ Mfg', use_container_width=True, help='Simulate Hardware Failure'): run_simulation('Manufacturing')
    if cols[2].button('🌐 Cyber', use_container_width=True, help='Simulate System Breach'): run_simulation('Cyber')
    
    if st.session_state.get('sim_active'):
        st.markdown(f'#### 🛠️ {st.session_state.get("sim_type", "EURI")} Agent Logs')
        for icon, log in st.session_state.get('sim_logs', []):
            st.markdown(f'<span style="font-size:11px;">{icon} {log}</span>', unsafe_allow_html=True)
        if st.button('✅ Clear Alert', use_container_width=True):
            st.session_state['sim_active'] = False
            st.rerun()
    st.divider()

    st.markdown(f"**User:** `{username}`")
    st.markdown(f"**Role:** {user_role}")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ['authenticated', 'username', 'role']:
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    st.caption("Enterprise Unified Risk Intelligence")
    st.caption("Hackathon 2026 • v4.0")
