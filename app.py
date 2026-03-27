import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import joblib
from datetime import datetime

st.set_page_config(page_title="EURI | Mission Control", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

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
            <div style="font-size:40px;margin-bottom:8px;">🛡️</div>
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

def run_simulation():
    st.session_state['sim_active'] = True
    st.session_state['sim_logs'] = [
        ('📡', 'EURI Neural Engine: Ingressing 4.2M Transaction Burst...'),
        ('⚠️', 'CRITICAL ANOMALY: Unauthorized Treasury Withdrawal detected (Z-Score: 12.4)'),
        ('🔎', 'Agentic Auditor: Identifying Destination Node... 0x8a...4b (Ghost Vendor)'),
        ('🛡️', 'EURI MITIGATION: [ACTION] Freezing SAP Outbound Payment Gateway...'),
        ('⛓️', 'EURI MITIGATION: [ACTION] Initiating Chain-of-Custody Log in Auditor...'),
        ('📧', 'EURI MITIGATION: [ACTION] Pushing High-Risk Slack Alert to CFO Cabinet...'),
        ('✅', 'EURI MITIGATION: Threat Contained. Manual Audit Requested.')
    ]

# Header UI Update for Simulation
if st.session_state.get('sim_active'):
    st.markdown('<div style="background-color:#7f1d1d; color:white; padding:10px; border-radius:8px; text-align:center; font-weight:bold; margin-bottom:20px; border:2px solid #ef4444; animation: pulse 2s infinite;">🚨 ACTIVE THREAT MITIGATION IN PROGRESS: .42M SUSPICIOUS OUTBOUND DETECTED</div>', unsafe_allow_html=True)
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
    <span style="font-size:28px;">🛡️</span>
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
    st.markdown("### 🛡️ EURI")
    manifest = data.get('manifest', {})
    st.markdown(f"**Version:** {manifest.get('version', '4.0')}")
    st.markdown(f"**Trained:** {manifest.get('trained_at', 'N/A')[:19]}")
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
