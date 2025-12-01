import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Picking Ops Dashboard",
    page_icon="📦",
    layout="wide"
)

# Custom CSS for a cleaner "Analyst" look
st.markdown("""
<style>
    .metric-container {
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LIVE DATA LOADING ---
# ttl=60 keeps data for 60 seconds, then re-fetches from Google Sheets
@st.cache_data(ttl=60)
def load_data():
    # Constructing the export URL from your Google Sheet ID
    sheet_id = "12yDda2q5hQ8RsVZAyFu13H-t7oEZ8GUMygUKtKmBVPg"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
    
    try:
        df = pd.read_csv(sheet_url)
        
        # --- DATA CLEANING ---
        # 1. Date Handling
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # 2. Numeric Conversion (Cleaning '%' and ',' symbols)
        numeric_cols = ['Total_picked', 'IRT Marked', 'IRT%']
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 3. Ensure IRT% is scaled correctly (if data is 0.45, it stays. if 45, it might be %)
        # Based on your data, it seems fine, but ensure calculated fields are used for accuracy.
        
        return df.sort_values(by="Date", ascending=False)
    
    except Exception as e:
        st.error(f"⚠️ Error connecting to Google Sheets. Please check permissions. Details: {e}")
        return pd.DataFrame() # Return empty DF to prevent crash

# Load the data
df = load_data()

# Stop if data failed to load
if df.empty:
    st.stop()

# --- 3. SIDEBAR CONTROLS (The Control Tower) ---
st.sidebar.title("🛠️ Control Tower")

# A. Manual Refresh
if st.sidebar.button('🔄 Refresh Live Data'):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# B. Date Filter
min_date = df['Date'].min()
max_date = df['Date'].max()

st.sidebar.subheader("📅 Timeframe")
date_range = st.sidebar.date_input(
    "Select Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply Date Filter
if len(date_range) == 2:
    start_date, end_date = date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
else:
    df_filtered = df.copy()

# C. Floor Filter
st.sidebar.subheader("🏢 Location Filters")
floor_list = ['All'] + sorted(df_filtered['Floor'].astype(str).unique().tolist())
selected_floor = st.sidebar.selectbox("Select Floor", floor_list)

if selected_floor != 'All':
    df_filtered = df_filtered[df_filtered['Floor'] == selected_floor]

# D. Zone Filter (Cascading)
zone_list = ['All'] + sorted(df_filtered['Picking Zone'].astype(str).unique().tolist())
selected_zone = st.sidebar.multiselect("Select Picking Zone(s)", zone_list, default='All')

if 'All' not in selected_zone:
    df_filtered = df_filtered[df_filtered['Picking Zone'].isin(selected_zone)]

# --- 4. MAIN DASHBOARD UI ---

st.title("📦 Picking Operations Dashboard")
st.markdown("### Black Belt Quality Analysis")

# --- ROW 1: EXECUTIVE KPIs ---
# Calculating aggregates on the fly based on filters
total_vol = df_filtered['Total_picked'].sum()
total_defects = df_filtered['IRT Marked'].sum()
avg_irt = (total_defects / total_vol * 100) if total_vol > 0 else 0

# Simple Sigma Estimation (Yield = 1 - Defect Rate)
process_yield = (1 - (total_defects / total_vol)) * 100 if total_vol > 0 else 0
if avg_irt < 0.5:
    status = "🟢 Optimized"
elif avg_irt < 1.0:
    status = "🟡 Warning"
else:
    status = "🔴 Critical"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Units Picked", f"{int(total_vol):,}")
c2.metric("Total IRT Errors", f"{int(total_defects)}")
c3.metric("Global Error Rate (IRT%)", f"{avg_irt:.2f}%", delta_color="inverse")
c4.metric("Process Health", status, help="Based on Error Rate Thresholds")

st.markdown("---")

# --- ROW 2: STRATEGIC ANALYSIS ---

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🎯 Volume vs. Quality Matrix")
    st.caption("Identify High Volume/High Error zones (Top Left quadrant)")
    
    # Group by Zone for Scatter Plot
    zone_metrics = df_filtered.groupby('Picking Zone').agg({
        'Total_picked': 'sum',
        'IRT Marked': 'sum'
    }).reset_index()
    zone_metrics['Calculated IRT%'] = (zone_metrics['IRT Marked'] / zone_metrics['Total_picked'] * 100).fillna(0)
    
    fig_scatter = px.scatter(
        zone_metrics,
        x="Total_picked",
        y="Calculated IRT%",
        size="Total_picked",
        color="Calculated IRT%",
        hover_name="Picking Zone",
        color_continuous_scale="RdYlGn_r", # Red = High Error, Green = Low Error
        labels={"Total_picked": "Volume", "Calculated IRT%": "Error Rate %"},
        height=400
    )
    # Add reference lines for context
    fig_scatter.add_hline(y=zone_metrics['Calculated IRT%'].mean(), line_dash="dash", line_color="gray", annotation_text="Avg Error")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("⚠️ Pareto: Top Defects")
    st.caption("80% of errors come from these zones")
    
    # Sort by raw defect count
    pareto_data = zone_metrics.sort_values(by="IRT Marked", ascending=False).head(10)
    
    fig_pareto = px.bar(
        pareto_data,
        x="IRT Marked",
        y="Picking Zone",
        orientation='h',
        color="IRT Marked",
        color_continuous_scale="Reds",
        height=400
    )
    fig_pareto.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_pareto, use_container_width=True)

# --- ROW 3: TEMPORAL & SPATIAL ANALYSIS ---
st.markdown("---")
c_trend, c_heat = st.columns(2)

with c_trend:
    st.subheader("📈 Trend Control Chart")
    daily_data = df_filtered.groupby('Date')[['Total_picked', 'IRT Marked']].sum().reset_index()
    daily_data['Daily IRT%'] = (daily_data['IRT Marked'] / daily_data['Total_picked'] * 100)
    
    fig_line = px.line(daily_data, x="Date", y="Daily IRT%", markers=True)
    fig_line.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Target")
    st.plotly_chart(fig_line, use_container_width=True)

with c_heat:
    st.subheader("🔥 Floor Heatmap")
    # Pivot data: Rows=Floor, Cols=Date, Values=Error Rate
    try:
        heatmap_df = df_filtered.pivot_table(
            index="Floor", 
            columns=df_filtered['Date'].dt.date, 
            values="IRT%", 
            aggfunc="mean"
        )
        fig_heat = px.imshow(
            heatmap_df,
            labels=dict(x="Date", y="Floor", color="Error %"),
            color_continuous_scale="RdYlGn_r",
            aspect="auto"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    except:
        st.info("Insufficient data for Heatmap with current filters.")

# --- 5. DATA EXPORT ---
with st.expander("📂 View Raw Data Source"):
    st.dataframe(df_filtered.style.format({"IRT%": "{:.2f}%"}))
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered CSV",
        csv,
        "filtered_picking_data.csv",
        "text/csv",
        key='download-csv'
    )
