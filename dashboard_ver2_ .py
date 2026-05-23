"""
CrisisLens - Live Streamlit Dashboard
=======================================

Run:
    pip install streamlit scikit-learn pandas numpy plotly openpyxl
    streamlit run dashboard_new.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from pipeline import (
    load_and_preprocess,
    train_hazard_model,
    train_evac_model,
    build_dp_snapshot,
    HAZARD_FEATURES,
    EVAC_FEATURES,
)

DATASET_PATH = "new_dataset.csv"
REFRESH_EVERY = 30

st.set_page_config(
    page_title="CrisisLens Live",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] div { color: #e2e8f0 !important; }

    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 1.2s infinite;
        margin-right: 8px;
        vertical-align: middle;
    }

    @keyframes pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50% { opacity:0.4; transform:scale(1.5); }
    }

    .ts-badge {
        background: #1e293b;
        color: #94a3b8;
        font-size: 13px;
        padding: 5px 14px;
        border-radius: 20px;
        font-family: monospace;
    }

    .sec-head {
        font-size: 15px;
        font-weight: 600;
        color: #1e293b;
        margin: 18px 0 8px;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="📂 Loading dataset...")
def get_data(path):
    return load_and_preprocess(path)


@st.cache_resource(show_spinner="🤖 Training AI models...")
def get_models(path):
    df = load_and_preprocess(path)
    h_model, h_mae, h_r2 = train_hazard_model(df)
    e_model, e_mae, e_r2 = train_evac_model(df)
    return h_model, e_model, h_mae, h_r2, e_mae, e_r2


try:
    df = get_data(DATASET_PATH)
    hazard_model, evac_model, h_mae, h_r2, e_mae, e_r2 = get_models(DATASET_PATH)
except FileNotFoundError:
    st.error(
        f"❌ Dataset not found: `{DATASET_PATH}`\n\n"
        "Make sure `new_dataset.csv` and `pipeline.py` are in the same folder."
    )
    st.stop()

timestamps = sorted(df["Timestamp"].unique())
total_ts = len(timestamps)

idx = (int(time.time()) // REFRESH_EVERY) % total_ts
curr_ts = timestamps[idx]
snapshot = df[df["Timestamp"] == curr_ts].copy()

with st.sidebar:
    st.markdown("## 🚨 CrisisLens")
    st.markdown("**Privacy-First Emergency AI**")
    st.markdown("---")

    progress = idx / max(total_ts - 1, 1)
    st.markdown("### 🔴 Live Feed")
    st.progress(progress)
    st.caption(f"Reading {idx + 1} of {total_ts} · refreshes every {REFRESH_EVERY}s")

    st.markdown("---")

    st.markdown("### 🔐 Differential Privacy")
    epsilon = st.slider(
        "ε (epsilon)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Lower ε = stronger privacy, more noise. Higher ε = less noise."
    )

    if epsilon <= 1.0:
        st.success(f"ε = {epsilon:.1f} · 🟢 Strong privacy")
    elif epsilon <= 3.0:
        st.warning(f"ε = {epsilon:.1f} · 🟡 Moderate privacy")
    else:
        st.error(f"ε = {epsilon:.1f} · 🔴 Weak privacy")

    st.caption(f"Laplace scale = {(1 / epsilon):.3f}")
    st.caption("Applies to: adults, children, elderly, disabled, total, hazard, evac")
    st.caption("Gaussian δ=1e-5: heatmap density")

    st.markdown("---")

    st.markdown("### 🤖 Model Performance")
    st.markdown(f"""
    **Hazard Score** *(Gradient Boosting)*
    - MAE : `{h_mae:.4f}`
    - R²  : `{h_r2:.4f}`

    **Evac Priority** *(Random Forest)*
    - MAE : `{e_mae:.4f}`
    - R²  : `{e_r2:.4f}`
    """)

    st.markdown("---")
    st.markdown("### 🏢 Zones")
    st.markdown("""
    - **Z1** Reception · Level 1
    - **Z2** Workspace · Level 1
    - **Z3** MeetingRoom · Level 1
    - **Z4** Pantry · Level 1
    - **Z5** Corridor · Level 2
    - **Z6** ExitArea · Level 2
    """)

np.random.seed(int(time.time()) // REFRESH_EVERY)
dp_df = build_dp_snapshot(snapshot, hazard_model, evac_model, epsilon)

col_title, col_ts, col_next = st.columns([3, 2, 1])

with col_title:
    st.markdown(
        '<span class="live-dot"></span>'
        '<span style="font-size:22px; font-weight:700; color:#1e293b;">'
        'CrisisLens — Live Warden Dashboard</span>',
        unsafe_allow_html=True
    )
    st.caption("Differential privacy active · No personal data exposed to wardens")

with col_ts:
    st.markdown(
        f'<br><span class="ts-badge">'
        f'📡 {pd.Timestamp(curr_ts).strftime("%d %b %Y  %H:%M:%S")}'
        f'</span>',
        unsafe_allow_html=True
    )

with col_next:
    secs_left = REFRESH_EVERY - (int(time.time()) % REFRESH_EVERY)
    st.markdown(
        f"<br><small style='color:#94a3b8;'>Next update in<br>"
        f"<b style='font-size:18px; color:#1e293b;'>{secs_left}s</b></small>",
        unsafe_allow_html=True
    )

st.divider()

# KPI METRICS
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
high = (dp_df["Label"] == "High").sum()

with k1:
    st.metric("👥 Total Occupancy", dp_df["DP_Occupancy"].sum())
with k2:
    st.metric("🧑 Adults", dp_df["DP_Adult"].sum())
with k3:
    st.metric("👧 Children", dp_df["DP_Child"].sum())
with k4:
    st.metric("👴 Elderly", dp_df["DP_Elderly"].sum())
with k5:
    st.metric("♿ Disabled", dp_df["DP_Disabled"].sum())
with k6:
    st.metric("🔥 Avg Hazard", f"{dp_df['DP_Hazard'].mean():.2f}")
with k7:
    st.metric(
        "🔴 High Hazard Zones",
        high,
        delta=f"{high} alert" if high > 0 else None,
        delta_color="inverse"
    )

st.divider()

# ZONE CARDS
st.markdown('<p class="sec-head">🏢 Zone-by-Zone Status</p>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
cols = [col_a, col_b, col_c]

for i, (_, zone) in enumerate(dp_df.iterrows()):
    with cols[i % 3]:
        with st.container(border=True):

            left, right = st.columns([3, 1])

            with left:
                st.markdown(
                    f"**{zone['Zone_ID']} — {zone['Zone_Name']}**  \n"
                    f"<small style='color:#94a3b8;'>{zone['Floor']}</small>",
                    unsafe_allow_html=True
                )

            with right:
                label = zone["Label"]
                txt_color = zone["TxtColor"]
                bg_color = zone["BgColor"]

                st.markdown(
                    f"<div style='text-align:center; margin-top:4px;'>"
                    f"<span style='background:{bg_color}; color:{txt_color}; "
                    f"border-radius:20px; padding:4px 12px; "
                    f"font-size:12px; font-weight:600;'>{label}</span></div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")

            st.markdown(
                "<small style='color:#64748b; font-weight:600;"
                "text-transform:uppercase; letter-spacing:0.05em;'>"
                "People (DP-noised)</small>",
                unsafe_allow_html=True
            )

            p1, p2, p3, p4 = st.columns(4)

            with p1:
                st.metric("🧑 Adults", zone["DP_Adult"])
            with p2:
                st.metric("👧 Children", zone["DP_Child"])
            with p3:
                st.metric("👴 Elderly", zone["DP_Elderly"])
            with p4:
                st.metric("♿ Disabled", zone["DP_Disabled"])

            st.markdown("---")

            m1, m2 = st.columns(2)

            with m1:
                st.metric("👥 Total (DP)", zone["DP_Occupancy"])
                st.metric("🚨 Evac Score (DP)", zone["DP_Evac"])

            with m2:
                st.metric("🔥 Hazard (DP)", zone["DP_Hazard"])
                st.metric("🌡 Temperature", f"{zone['Temperature']}°C")

            e1, e2, e3 = st.columns(3)

            with e1:
                st.metric("💨 CO₂", f"{zone['CO2']} ppm")
            with e2:
                st.metric("💧 Humidity", f"{zone['Humidity']}%")
            with e3:
                st.metric("🌫 PM2.5", zone["PM25"])

            fill_pct = min(100, (zone["DP_Hazard"] / 2.82) * 100)
            st.progress(
                fill_pct / 100,
                text=f"Hazard level: {zone['DP_Hazard']} / 2.82"
            )
            st.caption(f"⚠ Laplace-noised · ε = {epsilon:.1f}")

st.divider()

# PEOPLE BREAKDOWN CHART
st.markdown(
    '<p class="sec-head">👥 People Breakdown by Zone (DP counts)</p>',
    unsafe_allow_html=True
)

fig_people = go.Figure()

fig_people.add_trace(go.Bar(
    x=dp_df["Zone_ID"],
    y=dp_df["DP_Adult"],
    name="Adults",
    marker_color="#3b82f6"
))

fig_people.add_trace(go.Bar(
    x=dp_df["Zone_ID"],
    y=dp_df["DP_Child"],
    name="Children",
    marker_color="#60a5fa"
))

fig_people.add_trace(go.Bar(
    x=dp_df["Zone_ID"],
    y=dp_df["DP_Elderly"],
    name="Elderly",
    marker_color="#7c3aed"
))

fig_people.add_trace(go.Bar(
    x=dp_df["Zone_ID"],
    y=dp_df["DP_Disabled"],
    name="Disabled",
    marker_color="#0891b2"
))

fig_people.update_layout(
    barmode="stack",
    title=f"Stacked People Count per Zone — DP-noised ε = {epsilon:.1f}",
    height=320,
    margin=dict(t=40, b=20, l=20, r=20),
    legend=dict(orientation="h", y=-0.2),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig_people, width="stretch")

st.divider()

# LIVE CHARTS
st.markdown('<p class="sec-head">📈 Live Charts</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    fig_occ = go.Figure()

    fig_occ.add_trace(go.Bar(
        x=dp_df["Zone_ID"],
        y=dp_df["True_Occ"],
        name="True occupancy",
        marker_color="#94a3b8",
        opacity=0.5
    ))

    fig_occ.add_trace(go.Bar(
        x=dp_df["Zone_ID"],
        y=dp_df["DP_Occupancy"],
        name="DP occupancy",
        marker_color="#3b82f6"
    ))

    fig_occ.update_layout(
        title="Total Occupancy — True vs DP",
        barmode="group",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.3),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_occ, width="stretch")

with c2:
    fig_hz = go.Figure(go.Bar(
        x=dp_df["Zone_ID"],
        y=dp_df["DP_Hazard"],
        marker_color=[z["TxtColor"] for _, z in dp_df.iterrows()],
        text=dp_df["Label"],
        textposition="outside"
    ))

    fig_hz.update_layout(
        title="DP Hazard Score by Zone (0 – 2.82)",
        height=300,
        yaxis=dict(range=[0, 3.2]),
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_hz, width="stretch")

c3, c4 = st.columns(2)

with c3:
    fig_evac = px.bar(
        dp_df,
        x="Zone_ID",
        y="DP_Evac",
        color="Label",
        color_discrete_map={
            "Low": "#22c55e",
            "Medium": "#f97316",
            "High": "#ef4444"
        },
        title="DP Evacuation Priority by Zone",
        text_auto=True
    )

    fig_evac.update_layout(
        height=300,
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_evac, width="stretch")

with c4:
    categories = ["Temp", "PM2.5", "CO2", "Humidity", "VOC"]
    fig_radar = go.Figure()

    for _, zone in dp_df.iterrows():
        vals = [
            zone["Temperature"] / 25.3,
            zone["PM25"] / 17.36,
            zone["CO2"] / 670.57,
            zone["Humidity"] / 53.94,
            zone["VOC"] / 168.07,
        ]

        vals += [vals[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            name=zone["Zone_ID"],
            fill="toself",
            opacity=0.35
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Sensor Readings Normalised",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig_radar, width="stretch")

st.divider()

# HISTORICAL TREND
st.markdown(
    '<p class="sec-head">📉 Historical Trend — Last 10 Readings</p>',
    unsafe_allow_html=True
)

hist_idx = max(0, idx - 9)
hist_ts = timestamps[hist_idx: idx + 1]
hist_df = df[df["Timestamp"].isin(hist_ts)].copy()

hist_df["pred_hazard"] = np.clip(
    hazard_model.predict(hist_df[HAZARD_FEATURES]),
    0,
    2.82
)

hist_df["pred_evac"] = evac_model.predict(hist_df[EVAC_FEATURES])

h1, h2 = st.columns(2)

with h1:
    fig_hist_occ = px.line(
        hist_df.sort_values("Timestamp"),
        x="Timestamp",
        y="Total_Occupancy_Raw",
        color="Zone_ID",
        title="Occupancy Trend — Last 10 readings",
        markers=True
    )

    fig_hist_occ.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3)
    )

    st.plotly_chart(fig_hist_occ, width="stretch")

with h2:
    fig_hist_hz = px.line(
        hist_df.sort_values("Timestamp"),
        x="Timestamp",
        y="pred_hazard",
        color="Zone_ID",
        title="Hazard Score Trend — Last 10 readings",
        markers=True
    )

    fig_hist_hz.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3)
    )

    st.plotly_chart(fig_hist_hz, width="stretch")

st.divider()

# FULL DP OUTPUT TABLE
st.markdown(
    '<p class="sec-head">🔐 Full Privacy-Safe Output Table</p>',
    unsafe_allow_html=True
)

display_df = dp_df[[
    "Zone_ID",
    "Zone_Name",
    "Floor",
    "DP_Adult",
    "DP_Child",
    "DP_Elderly",
    "DP_Disabled",
    "DP_Occupancy",
    "DP_Hazard",
    "DP_Evac",
    "DP_Heatmap",
    "Label"
]].rename(columns={
    "Zone_ID": "Zone",
    "Zone_Name": "Name",
    "Floor": "Level",
    "DP_Adult": "Adults (DP)",
    "DP_Child": "Children (DP)",
    "DP_Elderly": "Elderly (DP)",
    "DP_Disabled": "Disabled (DP)",
    "DP_Occupancy": "Total (DP)",
    "DP_Hazard": "Hazard Score (DP)",
    "DP_Evac": "Evac Priority (DP)",
    "DP_Heatmap": "Heatmap (DP)",
    "Label": "Hazard Level"
})

st.dataframe(display_df, width="stretch", hide_index=True)

st.caption(
    f"⚠ All values Laplace/Gaussian noised at ε = {epsilon:.1f}. "
    f"Sensitivity: counts = 1, hazard = 2.82, evac = 53.54, heatmap = 1 Gaussian δ=1e-5. "
    f"No raw personal data is ever exposed to wardens."
)

time.sleep(1)
st.rerun()