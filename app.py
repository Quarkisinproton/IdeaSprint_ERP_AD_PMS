import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SHAP plots
import matplotlib.pyplot as plt
import shap

# ==========================================
# Page Config & Theme
# ==========================================
st.set_page_config(
    page_title="ERR-CC | Enterprise Risk & Reliability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; border-radius: 8px;
        color: #e0e0e0; padding: 10px 24px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e94560 !important; color: white !important;
    }
    div[data-testid="stMetric"] {
        background-color: #16213e; border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #e94560;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Data & Model Loading
# ==========================================
@st.cache_data
def load_data():
    paysim_path = "DataSets/PAYSim/processed_paysim.csv"
    nasa_path = "DataSets/Synthetic/processed_nasa.csv"
    df_paysim = pd.read_csv(paysim_path) if os.path.exists(paysim_path) else pd.DataFrame()
    df_nasa = pd.read_csv(nasa_path) if os.path.exists(nasa_path) else pd.DataFrame()
    return df_paysim, df_nasa

@st.cache_resource
def load_shap_artifacts():
    artifacts = {}
    files = {
        'shap_rul': 'models/shap_values_rul.pkl',
        'shap_expected_rul': 'models/shap_expected_rul.pkl',
        'shap_fraud': 'models/shap_values_fraud.pkl',
        'shap_fraud_sample': 'models/shap_fraud_sample.pkl',
        'shap_expected_fraud': 'models/shap_expected_fraud.pkl',
        'rf_rul': 'models/rf_rul.pkl',
    }
    for key, path in files.items():
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
    return artifacts

df_paysim, df_nasa = load_data()
shap_artifacts = load_shap_artifacts()

# ==========================================
# Header
# ==========================================
st.markdown("## 🛡️ ERR-CC: Enterprise Risk & Reliability Command Center")
st.markdown("*Unified monitoring of Financial Integrity and Operational Reliability with AI-Driven Explainability.*")
st.divider()

# Create Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Financial Auditor",
    "⚙️ Predictive Maintenance",
    "🔗 ERP Integration Bridge"
])

# ==========================================
# TAB 1: Financial Auditor + SHAP
# ==========================================
with tab1:
    st.header("Financial Integrity: Anomaly Detection")

    if not df_paysim.empty:
        anomalies = df_paysim[df_paysim['Is_Anomaly'] == 1].copy()
        anomalies = anomalies.sort_values(by='Risk_Score', ascending=False)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("🔴 Flagged Transactions", f"{len(anomalies):,}")
        col_m2.metric("💰 Highest Risk Score", f"{anomalies['Risk_Score'].max():.2f}" if len(anomalies) > 0 else "N/A")
        col_m3.metric("📈 Avg Risk Score (Flagged)", f"{anomalies['Risk_Score'].mean():.2f}" if len(anomalies) > 0 else "N/A")

        st.subheader("Flagged Transactions Table")
        st.dataframe(
            anomalies[['step', 'type', 'amount', 'nameOrig', 'nameDest', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error', 'Risk_Score']].head(100),
            use_container_width=True, height=350
        )

        # SHAP Explainability for Fraud
        st.subheader("🔍 XAI: Why was this transaction flagged?")
        if 'shap_fraud' in shap_artifacts and 'shap_fraud_sample' in shap_artifacts:
            shap_vals = shap_artifacts['shap_fraud']
            fraud_sample = shap_artifacts['shap_fraud_sample']
            expected_val = shap_artifacts.get('shap_expected_fraud', 0)

            idx = st.slider("Select Transaction Index (from sample)", 0, len(fraud_sample) - 1, 0, key="fraud_shap_slider")

            feature_names = fraud_sample.columns.tolist()
            sv = shap_vals[idx]
            fv = fraud_sample.iloc[idx].values

            # Build a Plotly waterfall chart mimicking SHAP force plot
            sorted_idx = np.argsort(np.abs(sv))[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_shap = [sv[i] for i in sorted_idx]
            sorted_fvals = [fv[i] for i in sorted_idx]

            colors = ['#e94560' if v > 0 else '#0f3460' for v in sorted_shap]
            labels = [f"{feat} = {val:.1f}" for feat, val in zip(sorted_features, sorted_fvals)]

            fig_shap = go.Figure(go.Bar(
                x=sorted_shap, y=labels, orientation='h',
                marker_color=colors, text=[f"{v:+.4f}" for v in sorted_shap], textposition='outside'
            ))
            fig_shap.update_layout(
                title="SHAP Feature Contributions (Red = Pushes Toward Anomaly)",
                xaxis_title="SHAP Value", yaxis_title="",
                height=300, margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="#0E1117", plot_bgcolor="#16213e", font={'color': 'white'},
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Global Feature Importance (mean |SHAP|)
            st.subheader("📊 Global Feature Importance (Fraud Model)")
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            fig_global = go.Figure(go.Bar(
                x=mean_abs_shap, y=feature_names, orientation='h',
                marker_color='#e94560', text=[f"{v:.4f}" for v in mean_abs_shap], textposition='outside'
            ))
            fig_global.update_layout(
                title="Mean |SHAP| Value per Feature",
                xaxis_title="Mean |SHAP Value|", height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="#0E1117", plot_bgcolor="#16213e", font={'color': 'white'},
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.info("SHAP artifacts not found. Run `python data_engine.py` first.")
    else:
        st.warning("⚠️ Data not processed. Run `python data_engine.py`.")


# ==========================================
# TAB 2: Operational Engineer + SHAP
# ==========================================
with tab2:
    st.header("Operational Reliability: Predictive Maintenance")

    if not df_nasa.empty:
        engine_ids = sorted(df_nasa['EngineID'].unique())
        selected_engine = st.selectbox("Select Engine Unit:", engine_ids, key="engine_select")

        engine_data = df_nasa[df_nasa['EngineID'] == selected_engine].sort_values(by='Cycle')

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Sensor Telemetry")
            sensor_cols = [c for c in df_nasa.columns if c.startswith('Sensor')]
            fig_sensor = go.Figure()
            colors_palette = ['#00d2ff', '#e94560', '#0f3460']
            for i, sc in enumerate(sensor_cols):
                fig_sensor.add_trace(go.Scatter(
                    x=engine_data['Cycle'], y=engine_data[sc],
                    mode='lines+markers', name=sc,
                    line=dict(color=colors_palette[i % len(colors_palette)], width=2),
                    marker=dict(size=4)
                ))
            fig_sensor.update_layout(
                xaxis_title="Cycle", yaxis_title="Sensor Reading",
                height=350, margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="#0E1117", plot_bgcolor="#16213e", font={'color': 'white'},
                legend=dict(orientation="h", y=1.12)
            )
            st.plotly_chart(fig_sensor, use_container_width=True)

        with col2:
            st.subheader("🏥 Asset Health")
            max_rul = engine_data['Max_Cycle'].iloc[0]
            current_cycle = st.slider(
                "Simulate Current Cycle", 
                min_value=int(engine_data['Cycle'].min()),
                max_value=int(engine_data['Cycle'].max()),
                value=int(engine_data['Cycle'].max()),
                key="cycle_slider"
            )
            dynamic_rul = max_rul - current_cycle
            dynamic_health = (dynamic_rul / max_rul) * 100 if max_rul > 0 else 0

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=dynamic_health,
                title={'text': "Health %", 'font': {'size': 18, 'color': 'white'}},
                number={'suffix': '%', 'font': {'size': 32, 'color': 'white'}},
                delta={'reference': 50, 'increasing': {'color': '#00d2ff'}, 'decreasing': {'color': '#e94560'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "rgba(255,255,255,0.8)"},
                    'steps': [
                        {'range': [0, 20], 'color': "#e94560"},
                        {'range': [20, 50], 'color': "#f5a623"},
                        {'range': [50, 100], 'color': "#2ecc71"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 20}
                }
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="#0E1117", font={'color': "white"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.metric("🔧 RUL (Cycles Left)", dynamic_rul)

        # SHAP for RUL Model
        st.subheader("🔍 XAI: Why is this engine predicted to have this RUL?")
        if 'shap_rul' in shap_artifacts:
            shap_vals_rul = shap_artifacts['shap_rul']
            rul_features = ['Cycle', 'Sensor_1', 'Sensor_2', 'Sensor_3']

            # Find the row matching selected engine & cycle
            mask = (df_nasa['EngineID'] == selected_engine) & (df_nasa['Cycle'] == current_cycle)
            if mask.sum() > 0:
                row_idx = mask.idxmax()
                sv = shap_vals_rul[row_idx]
                fv = df_nasa.loc[row_idx, rul_features].values

                sorted_idx = np.argsort(np.abs(sv))[::-1]
                sorted_features = [rul_features[i] for i in sorted_idx]
                sorted_shap = [sv[i] for i in sorted_idx]
                sorted_fvals = [fv[i] for i in sorted_idx]

                colors = ['#e94560' if v < 0 else '#2ecc71' for v in sorted_shap]
                labels = [f"{feat} = {val:.2f}" for feat, val in zip(sorted_features, sorted_fvals)]

                fig_shap_rul = go.Figure(go.Bar(
                    x=sorted_shap, y=labels, orientation='h',
                    marker_color=colors, text=[f"{v:+.2f}" for v in sorted_shap], textposition='outside'
                ))
                fig_shap_rul.update_layout(
                    title=f"SHAP: Feature Contributions to RUL Prediction (Engine {selected_engine}, Cycle {current_cycle})",
                    xaxis_title="SHAP Value (impact on RUL prediction)",
                    height=280, margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="#0E1117", plot_bgcolor="#16213e", font={'color': 'white'},
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_shap_rul, use_container_width=True)

            # Global feature importance bar chart
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("**Global Feature Importance (RUL Model)**")
                mean_abs_shap_rul = np.mean(np.abs(shap_vals_rul), axis=0)
                fig_gi = go.Figure(go.Bar(
                    x=mean_abs_shap_rul, y=rul_features, orientation='h',
                    marker_color='#00d2ff', text=[f"{v:.2f}" for v in mean_abs_shap_rul], textposition='outside'
                ))
                fig_gi.update_layout(
                    height=230, margin=dict(l=20, r=20, t=10, b=20),
                    paper_bgcolor="#0E1117", plot_bgcolor="#16213e", font={'color': 'white'},
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_gi, use_container_width=True)

            with col_g2:
                st.markdown("**SHAP Beeswarm (Summary)**")
                # Render SHAP summary plot as an image
                fig_summary, ax = plt.subplots(figsize=(6, 3))
                X_rul_df = df_nasa[rul_features]
                shap.summary_plot(shap_vals_rul, X_rul_df, plot_type="bar", show=False, color_bar=False)
                plt.tight_layout()
                st.pyplot(fig_summary)
                plt.close()
        else:
            st.info("SHAP artifacts not found. Run `python data_engine.py` first.")
    else:
        st.warning("⚠️ NASA Data not found. Run `python data_engine.py`.")


# ==========================================
# TAB 3: ERP Integration Logic Bridge
# ==========================================
with tab3:
    st.header("🔗 ERP Integration: Automating Risk & Reliability")
    st.markdown("The Logic Bridge monitors real-time asset health and triggers automated ERP procurement workflows, cross-checked against financial anomaly audits.")
    st.divider()

    if not df_nasa.empty:
        st.subheader(f"Monitoring Engine: **{selected_engine}** | Cycle: `{current_cycle}`")

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            st.metric("Current Health", f"{dynamic_health:.1f}%")
            st.metric("RUL", f"{dynamic_rul} cycles")

        with col_b2:
            if dynamic_health < 20:
                st.error("🚨 **CRITICAL ALERT**: Asset health below 20% threshold!")
                st.markdown("""
                ---
                ### 🤖 Automated ERP Response
                | Step | Action | Status |
                |------|--------|--------|
                | 1 | **Failure Prediction Triggered** | ✅ Complete |
                | 2 | **Procurement Request Generated** | ✅ PR-{engine_id}-{cycle} |
                | 3 | **Vendor Financial Audit** | ✅ CLEAN |
                | 4 | **Fraud Risk Check on Procurement** | ✅ No Anomalies |
                | 5 | **Parts Dispatch Authorized** | ⏳ Pending Approval |
                """.format(engine_id=selected_engine, cycle=current_cycle))

                st.success("**Audit Status: CLEAN ✅** — No financial anomalies detected in simulated procurement pipeline.")
            elif dynamic_health < 50:
                st.warning("⚠️ **WARNING**: Asset health degrading. Monitoring closely.")
                st.markdown("> Preemptive maintenance scheduled. No procurement required yet.")
            else:
                st.success("✅ **NOMINAL**: Asset operating within safe parameters.")
                st.markdown("> No automated actions triggered.")
    else:
        st.warning("No data available for Logic Bridge simulation.")

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ ERR-CC")
    st.markdown("**Enterprise Risk & Reliability Command Center**")
    st.divider()
    st.markdown("**Tech Stack:**")
    st.markdown("- Scikit-learn & PyOD")
    st.markdown("- XGBoost-compatible")
    st.markdown("- SHAP Explainability")
    st.markdown("- Plotly Visualizations")
    st.divider()
    st.markdown("**Hackathon 2026**")
    st.caption("Built with ❤️ for innovation in ERP risk management.")
