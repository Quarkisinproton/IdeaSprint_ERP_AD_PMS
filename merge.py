import re

with open('app.py', 'r') as f:
    app_v4 = f.read()

with open('app_v3_backup.py', 'r') as f:
    app_v3 = f.read()

# 1. Update tabs array
tabs_v4_pattern = r'tab_overview,\s*tab_finance,\s*tab_maintenance,\s*tab_xai,\s*tab_auditor\s*=\s*st\.tabs\(\s*\[(.*?)\]\s*\)'
tabs_v4_replacement = """tab_overview, tab_finance, tab_maintenance, tab_erp, tab_auditor = st.tabs([
    "📡 Command Center",
    "💰 Financial Threat Intel",
    "⚙️ Predictive Maintenance",
    "🔗 ERP Procurement Bridge",
    "🤖 Agentic Auditor"
])"""
app_v4 = re.sub(tabs_v4_pattern, tabs_v4_replacement, app_v4, flags=re.DOTALL)

# 2. Extract v3 blocks starting from `with tab_maintenance:` to just before `with st.sidebar:`
start_v3 = app_v3.find('with tab_maintenance:')
end_v3 = app_v3.find('\nwith st.sidebar:', start_v3)
v3_blocks = app_v3[start_v3:end_v3].strip()

# 3. Insert into v4
start_v4 = app_v4.find('# ============================================================\n# TAB 3: PREDICTIVE MAINTENANCE')
end_v4 = app_v4.find('# ============================================================\n# SIDEBAR')

if start_v4 != -1 and end_v4 != -1:
    app_hybrid = app_v4[:start_v4] + v3_blocks + "\n\n" + app_v4[end_v4:]
    with open('app.py', 'w') as f:
        f.write(app_hybrid)
    print("Merged successfully!")
else:
    print(f"Error: start_v4={start_v4}, end_v4={end_v4}")
