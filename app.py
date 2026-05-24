"""
Live Warden Dashboard — Page 1: Overview
Privacy-First AI for Emergency Response
Centre for Technology and Infusion, La Trobe University

Data source: data/dashboard_model_output.csv  (produced by model_dp.py)

Install dependencies:
    pip install streamlit streamlit-autorefresh plotly pandas numpy

Run with:
    streamlit run app.py
"""

import base64

from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh


# ── SVG icon library (Bootstrap Icons, MIT) — 16×16 viewBox ──────────────────
def _i(path, w=14):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}" '
        f'viewBox="0 0 16 16" fill="currentColor" '
        f'style="vertical-align:-2px;flex-shrink:0;">{path}</svg>'
    )

ICONS = {
    "fire":     _i('<path d="M8 16c3.314 0 6-2 6-5.5 0-1.5-.5-4-2.5-6 .25 1.5-1.25 2-1.25 2C11 4 9 .5 6 0c.357 2 .5 4-2 6-1.25 1-2 2.729-2 4.5C2 14 4.686 16 8 16zm0-1c-1.657 0-3-1-3-2.75 0-.75.25-2 1.25-3C6.125 10 7 10.5 7 10.5c-.375-1.25.5-3.25 2-3.5-.179 1-.25 2 1 3 .625.5 1 1.364 1 2.25C11 14 9.657 15 8 15z"/>'),
    "wind":     _i('<path d="M12.5 2A2.5 2.5 0 0 1 15 4.5a2.5 2.5 0 0 1-2.5 2.5H.5a.5.5 0 0 1 0-1h12a1.5 1.5 0 1 0-1.5-1.5.5.5 0 0 1-1 0A2.5 2.5 0 0 1 12.5 2M5 6a2.5 2.5 0 0 1 0 5H.5a.5.5 0 0 1 0-1H5a1.5 1.5 0 0 0 0-3 .5.5 0 0 1 0-1m6.5 4a2.5 2.5 0 0 1 0 5H.5a.5.5 0 0 1 0-1h11a1.5 1.5 0 0 0 0-3 .5.5 0 0 1 0-1"/>'),
    "evac":     _i('<path fill-rule="evenodd" d="M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5z"/><path fill-rule="evenodd" d="M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0v-5z"/>'),
    "check":    _i('<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/><path d="M10.97 4.97a.235.235 0 0 0-.02.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-1.071-1.05z"/>'),
    "building": _i('<path d="M14.763.075A.5.5 0 0 1 15 .5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5V14h-1v1.5a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5V10a.5.5 0 0 1 .342-.474L6 7.64V4.5a.5.5 0 0 1 .276-.447l8-4a.5.5 0 0 1 .487.022z"/>'),
    "lock":     _i('<path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2zm3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2z"/>'),
    "bell":     _i('<path d="M8 16a2 2 0 0 0 2-2H6a2 2 0 0 0 2 2zM8 1.918l-.797.161A4.002 4.002 0 0 0 4 6c0 .628-.134 2.197-.459 3.742-.16.767-.376 1.566-.663 2.258h10.244c-.287-.692-.502-1.49-.663-2.258C12.134 8.197 12 6.628 12 6a4.002 4.002 0 0 0-3.203-3.92L8 1.917zM14.22 12c.223.447.481.801.78 1H1c.299-.199.557-.553.78-1C2.68 10.2 3 6.88 3 6c0-2.42 1.72-4.44 4.005-4.901a1 1 0 1 1 1.99 0A5.002 5.002 0 0 1 13 6c0 .88.32 4.2 1.22 6z"/>'),
    "droplet":  _i('<path d="M8 16a6 6 0 0 0 6-6c0-1.655-1.122-2.904-2.432-4.362C10.254 4.176 8.75 2.503 8 0c0 0-6 5.686-6 10a6 6 0 0 0 6 6zM6.646 4.646l.708.708c-.29.29-1.128 1.311-1.907 2.87l-.894-.448c.82-1.641 1.717-2.753 2.093-3.13z"/>'),
    "warning":  _i('<path d="M7.938 2.016A.13.13 0 0 1 8.002 2a.13.13 0 0 1 .063.016.146.146 0 0 1 .054.057l6.857 11.667c.036.06.035.124.002.183a.163.163 0 0 1-.054.06.116.116 0 0 1-.066.017H1.146a.115.115 0 0 1-.066-.017.163.163 0 0 1-.054-.06.176.176 0 0 1 .002-.183L7.884 2.073a.147.147 0 0 1 .054-.057zm1.044-.45a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566z"/><path d="M7.002 12a1 1 0 1 1 2 0 1 1 0 0 1-2 0zM7.1 5.995a.905.905 0 1 1 1.8 0l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995z"/>'),
    "alert":    _i('<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/><path d="M7.002 11a1 1 0 1 1 2 0 1 1 0 0 1-2 0zM7.1 4.995a.905.905 0 1 1 1.8 0l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 4.995z"/>'),
    "clock":    _i('<path d="M8 3.5a.5.5 0 0 0-1 0V9a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 8.71V3.5z"/><path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z"/>'),
}

# Flame SVG as base64 data URI — used as a Plotly layout image on the floor plan
_flame_svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<path fill="#842420" d="M8 16c3.314 0 6-2 6-5.5 0-1.5-.5-4-2.5-6 '
    '.25 1.5-1.25 2-1.25 2C11 4 9 .5 6 0c.357 2 .5 4-2 6-1.25 1-2 2.729-2 '
    '4.5C2 14 4.686 16 8 16zm0-1c-1.657 0-3-1-3-2.75 0-.75.25-2 1.25-3C6.125 '
    '10 7 10.5 7 10.5c-.375-1.25.5-3.25 2-3.5-.179 1-.25 2 1 3 .625.5 1 1.364 '
    '1 2.25C11 14 9.657 15 8 15z"/>'
    '</svg>'
)
FLAME_URI = "data:image/svg+xml;base64," + base64.b64encode(_flame_svg.encode()).decode()

# Floor map image as base64 data URI
try:
    _floor_img = Image.open("floor_map.png")
    import io as _io
    _buf = _io.BytesIO()
    _floor_img.save(_buf, format="PNG")
    FLOOR_MAP_URI = "data:image/png;base64," + base64.b64encode(_buf.getvalue()).decode()
except FileNotFoundError:
    FLOOR_MAP_URI = None


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Warden Dashboard",
    page_icon=":office:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh interval driven by speed selector ────────────────────────────
_speed_interval_ms = {"0.5×": 60_000, "1×": 30_000, "2×": 15_000, "4×": 5_000, "8×": 2_000}
_interval = _speed_interval_ms.get(st.session_state.get("speed", "1×"), 30_000)
refresh_count = st_autorefresh(interval=_interval, key="live_refresh")

# ── Custom CSS — clean academic light theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600&display=swap');

/* Earthy palette — #465030 green · #C0A425 mustard · #D0BCAD blush
                    #C08A80 terracotta · #B01808 brick · #842420 burgundy */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F0E8E2;
    color: #2C1810;
}
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
button[data-baseweb="tab"] { color: #842420 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #842420 !important; }
[data-baseweb="tab-highlight"] { background-color: #842420 !important; }
[data-baseweb="tab-border"] { background-color: #D0BCAD !important; }
[data-testid="block-container"],
[data-testid="stMainBlockContainer"] {
    padding-top: 0.5rem !important;
}
section.main > div:first-child { padding-top: 0 !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"], section.main, .main .block-container,
[data-testid="stExpander"], details, summary {
    background-color: #F0E8E2 !important;
    color: #2C1810 !important;
}
[data-testid="stSidebar"] {
    background-color: #FDFAF7;
    border-right: 1.5px solid #D0BCAD;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    font-size: 0.82rem;
    color: #7A6A5A;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-weight: 500;
}
h1 { font-family: 'Playfair Display', serif !important; font-size: 1.7rem !important; color: #2C1810 !important; }
h2 { font-family: 'DM Sans', sans-serif !important; font-size: 1.05rem !important; font-weight: 600 !important;
     color: #465030 !important; letter-spacing: 0.04em !important; text-transform: uppercase !important; }

.kpi-card {
    background: #FFFFFF;
    border: 1.5px solid #D0BCAD;
    border-radius: 10px;
    padding: 10px 14px 9px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
}
.kpi-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #A89880;
    margin-bottom: 3px;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #2C1810;
    line-height: 1;
}
.kpi-sub { font-size: 0.68rem; color: #A89880; margin-top: 3px; }

.zone-card {
    background: #FFFFFF;
    border: 1.5px solid #D0BCAD;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.zone-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.zone-id   { font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 700; color: #465030;
             background: #E5ECDE; padding: 1px 7px; border-radius: 5px; display: inline-block; }
.zone-name { font-size: 0.95rem; font-weight: 600; color: #2C1810; }
.zone-occ  { font-family: 'DM Mono', monospace; font-size: 1.5rem; font-weight: 500; color: #2C1810; }
.zone-occ-label { font-size: 0.68rem; color: #A89880; text-transform: uppercase; letter-spacing: 0.05em; }
.zone-pills { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }

.pill { font-size: 0.68rem; font-weight: 600; padding: 2px 8px; border-radius: 20px;
        letter-spacing: 0.04em; text-transform: uppercase; }
.pill-Low       { background: #E5ECDE; color: #465030; }
.pill-Moderate  { background: #F5EDD0; color: #8A6E15; }
.pill-High      { background: #F5DED8; color: #B01808; }
.pill-Critical  { background: #EDD5D3; color: #842420; }
.pill-Cleared   { background: #E5ECDE; color: #465030; }
.pill-Nearly    { background: #F5EDD0; color: #8A6E15; }
.pill-Occupied  { background: #F5DED8; color: #B01808; }
.pill-NotStarted{ background: #EDE5DC; color: #7A6A5A; }
.pill-Yes       { background: #E5ECDE; color: #465030; }
.pill-Partial   { background: #F5EDD0; color: #8A6E15; }
.pill-No        { background: #EDD5D3; color: #842420; }
.pill-alarm     { background: #EDD5D3; color: #842420; }
.pill-sprinkler { background: #DDE8F0; color: #2A5A80; }

.alert-banner {
    background: #EDD5D3;
    border: 1.5px solid #C8A0A0;
    border-left: 5px solid #842420;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.alert-banner-icon { font-size: 1.4rem; }
.alert-banner-text { font-size: 0.88rem; color: #5A1A18; font-weight: 500; }
.alert-banner-bold { font-weight: 700; }

.routine-banner {
    background: #E5ECDE;
    border: 1.5px solid #AACBA0;
    border-left: 5px solid #465030;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 18px;
}
.routine-banner p { font-size: 0.88rem; color: #2A3A1A; margin: 0; font-weight: 500; }

.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #B0A090;
    border-bottom: 1.5px solid #D0BCAD;
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 6px;
}

.event-log-container {
    background: #FFFFFF;
    border: 1.5px solid #D0BCAD;
    border-radius: 10px;
    padding: 14px 16px;
    max-height: 280px;
    overflow-y: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}
.event-row { padding: 4px 0; border-bottom: 1px solid #EDE5DC; color: #465030; }
.event-row:last-child { border-bottom: none; }
.event-time { color: #B0A090; margin-right: 10px; }
.event-type-FE { color: #842420; font-weight: 600; }
.event-type-SS { color: #B01808; font-weight: 600; }
.event-type-EV { color: #8A6E15; }
.event-type-RM { color: #7A6A5A; }

.dp-notice {
    background: #EDE5DC;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.75rem;
    color: #7A6A5A;
    margin-top: 10px;
    border-left: 3px solid #C08A80;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EDD5D3;
    border: 1px solid #C8A0A0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #842420;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.live-dot {
    width: 7px; height: 7px;
    background: #842420;
    border-radius: 50%;
    animation: pulse 1.4s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

@keyframes glow-pulse {
    0%, 100% { box-shadow: none; }
    50%       { box-shadow: 0 0 18px 5px rgba(132,36,32,0.28); border-color: #842420 !important; }
}
.alert-banner-fire { animation: glow-pulse 2.2s ease-in-out infinite; }

@keyframes kpi-flash {
    0%, 100% { box-shadow: none; }
    50%       { box-shadow: 0 0 10px rgba(132,36,32,0.22); }
}
.kpi-card-critical { animation: kpi-flash 2s ease-in-out infinite; }

.stButton > button {
    background: #465030;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 18px;
}
.stButton > button:hover { background: #5A6840; }
div[data-testid="metric-container"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/dashboard_model_output.csv"

ZONE_META = {
    "Z1": "Reception",
    "Z2": "Workspace",
    "Z3": "Meeting Room",
    "Z4": "Pantry",
    "Z5": "Corridor",
    "Z6": "Exit Area",
}

FIRE_SCENARIOS = {
    "INC-20260401-F1": {"origin": "Z3", "level": "Level 1", "label": "Fire Incident — 1 Apr 2026 (Level 1)"},
    "INC-20260402-F2": {"origin": "Z5", "level": "Level 2", "label": "Fire Incident — 2 Apr 2026 (Level 2)"},
}
ROUTINE_SCENARIOS = {
    "MON-20260401": "Routine Monitoring — 1 Apr 2026 (Level 1)",
    "MON-20260402": "Routine Monitoring — 2 Apr 2026 (Level 1)",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_data
def load_heatmap_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def get_base_incident(incident_id: str) -> str:
    for base in list(FIRE_SCENARIOS.keys()) + list(ROUTINE_SCENARIOS.keys()):
        if incident_id == base or incident_id.startswith(base + "-"):
            return base
    return incident_id


def filter_scenario(df: pd.DataFrame, base_id: str) -> pd.DataFrame:
    mask = df["Incident_ID"].apply(lambda x: get_base_incident(str(x)) == base_id)
    return df[mask].copy()


# ── Session state ─────────────────────────────────────────────────────────────
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = None
if "paused" not in st.session_state:
    st.session_state.paused = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = -1
if "speed" not in st.session_state:
    st.session_state.speed = "1×"
if "heatmap_fullscreen" not in st.session_state:
    st.session_state.heatmap_fullscreen = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Warden Dashboard")
    st.markdown("<p style='color:#A89880;font-size:0.75rem;margin-top:-8px;'>Privacy-First AI · La Trobe CTI</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    scenario_options = {
        **{v["label"]: k for k, v in FIRE_SCENARIOS.items()},
        **{v: k for k, v in ROUTINE_SCENARIOS.items()},
    }
    selected_label = st.selectbox("Scenario", list(scenario_options.keys()))
    selected_base  = scenario_options[selected_label]

    st.markdown("---")
    st.markdown("**Privacy Budget (ε)**")
    epsilon = st.slider(
        "Epsilon", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Lower ε = stronger privacy (more Laplace noise). Higher ε = less noise, higher utility.",
        label_visibility="collapsed",
    )
    privacy_label = (
        "🔒 Strong privacy — expect visible noise" if epsilon < 1.0
        else ("Balanced privacy–utility" if epsilon <= 2.0
              else "📊 High utility — reduced privacy")
    )
    st.markdown(f"""
    <div class='dp-notice'>
        ε = <b>{epsilon:.1f}</b> · Laplace scale = <b>{1/epsilon:.2f}</b><br>{privacy_label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Live Feed**")

    speed = st.select_slider(
        "Speed", options=["0.5×", "1×", "2×", "4×", "8×"], value=st.session_state.speed,
        label_visibility="visible",
    )
    st.session_state.speed = speed

    # Pause / Resume toggle
    pause_label = "Pause Auto-Advance" if not st.session_state.paused else "Resume Auto-Advance"
    if st.button(pause_label):
        st.session_state.paused = not st.session_state.paused

    # Manual reset
    if st.button("Restart from Beginning"):
        st.session_state.tick = 0
        st.session_state.paused = False

    speed_interval_ms = {"0.5×": 60_000, "1×": 30_000, "2×": 15_000, "4×": 5_000, "8×": 2_000}
    interval_ms = speed_interval_ms[speed]
    st.markdown(
        f"<p style='font-size:0.7rem;color:#CCCCDD;margin-top:8px;'>"
        f"Auto-advances every {interval_ms // 1000} s.<br>"
        f"Each step = 5 min of simulated time.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem;color:#CCCCDD;'>"
        "Occupancy figures are DP-protected.<br>"
        "No personal identifiers are displayed.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#FFFFFF;text-transform:uppercase;"
        "letter-spacing:0.08em;font-weight:700;margin-bottom:4px;'>Visual Overlays</p>",
        unsafe_allow_html=True,
    )
    map_opacity    = st.slider("Map Background Opacity", 0.1, 1.0, 0.88, step=0.05, key="map_opacity")
    heat_intensity = st.slider("Fire Hazard Intensity",  0.1, 1.0, 0.35, step=0.05, key="heat_intensity")
    bubble_scale   = st.slider("Density Bubble Size Boost", 0.3, 2.0, 0.8, step=0.05, key="bubble_scale")


# ── Load & filter ─────────────────────────────────────────────────────────────
try:
    df_all = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Run `model_dp.py` first.")
    st.stop()

# Reset tick when scenario changes
if st.session_state.last_scenario != selected_base:
    st.session_state.tick = 0
    st.session_state.paused = False
    st.session_state.last_scenario = selected_base

df_scenario = filter_scenario(df_all, selected_base)
if df_scenario.empty:
    st.error("No data found for this scenario.")
    st.stop()

timestamps = sorted(df_scenario["Timestamp"].unique())
n_ticks    = len(timestamps)

# ── Auto-advance on each 30s refresh (unless paused or at end) ────────────────
if refresh_count != st.session_state.last_refresh:
    st.session_state.last_refresh = refresh_count
    if not st.session_state.paused and st.session_state.tick < n_ticks - 1:
        st.session_state.tick += 1


current_ts = timestamps[st.session_state.tick]
df_now     = df_scenario[df_scenario["Timestamp"] == current_ts].copy()


# ── Re-apply DP noise with current epsilon ────────────────────────────────────
def laplace_noise(series: pd.Series, eps: float) -> pd.Series:
    scale = 1.0 / max(eps, 0.01)
    noisy = series.astype(float) + np.random.laplace(0, scale, len(series))
    return noisy.clip(0).round().astype(int)

np.random.seed(int(st.session_state.tick * 1000 + epsilon * 100))
df_now = df_now.copy()

# Apply DP noise only to the total — derive sub-counts proportionally so they
# can never exceed the noised total (fixes the composition problem).
df_now["DP_Total_Occupancy"] = laplace_noise(df_now["Total_Occupancy_Raw"], epsilon)
raw_total = df_now["Total_Occupancy_Raw"].clip(lower=1)  # avoid div-by-zero
dp_total  = df_now["DP_Total_Occupancy"]
df_now["DP_Adult_Count"]    = (df_now["Adult_Count"]   / raw_total * dp_total).clip(0).round().astype(int)
df_now["DP_Child_Count"]    = (df_now["Child_Count"]   / raw_total * dp_total).clip(0).round().astype(int)
df_now["DP_Elderly_Count"]  = (df_now["Elderly_Count"] / raw_total * dp_total).clip(0).round().astype(int)
df_now["DP_Disabled_Count"] = (df_now["Disabled_Count"]/ raw_total * dp_total).clip(0).round().astype(int)
df_now["DP_Vulnerable_Count"] = (
    df_now["DP_Child_Count"] + df_now["DP_Elderly_Count"] + df_now["DP_Disabled_Count"]
)

# ── Load & filter heatmap cluster data ───────────────────────────────────────
try:
    _df_heatmap_all = load_heatmap_data("data/dp_dashboard_heatmap.csv")
    _df_heatmap_all = filter_scenario(_df_heatmap_all, selected_base)
    df_heatmap = _df_heatmap_all[_df_heatmap_all["Timestamp"] == current_ts].copy()
except Exception:
    df_heatmap = pd.DataFrame()


# ── DP Noise Tracker card (sidebar, password-protected) ──────────────────────
_raw_sum = int(df_now["Total_Occupancy_Raw"].sum())
_dp_sum  = int(df_now["DP_Total_Occupancy"].sum())
_diff    = _dp_sum - _raw_sum
_diff_str   = f"+{_diff}" if _diff > 0 else str(_diff)
_diff_color = "#E8A87C" if _diff > 0 else ("#D9534F" if _diff < 0 else "#6BAE8E")
_diff_label = "over-reported" if _diff > 0 else ("under-reported" if _diff < 0 else "exact match")

if "noise_tracker_unlocked" not in st.session_state:
    st.session_state.noise_tracker_unlocked = False

with st.sidebar:
    st.markdown("---")
    st.markdown("**DP Noise Tracker**")

    if st.session_state.noise_tracker_unlocked:
        st.markdown(f"""
        <div class='dp-notice' style='padding:12px 14px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <span style='color:#A89880;font-size:0.72rem;'>Raw Count</span>
                <span style='font-weight:600;font-size:0.9rem;'>{_raw_sum}</span>
            </div>
            <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                <span style='color:#A89880;font-size:0.72rem;'>DP Count</span>
                <span style='font-weight:600;font-size:0.9rem;'>{_dp_sum}</span>
            </div>
            <div style='border-top:1px solid #C4B8A8;padding-top:8px;display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:#A89880;font-size:0.72rem;'>Noise Applied</span>
                <span style='font-weight:700;font-size:1.05rem;color:{_diff_color};'>{_diff_str}</span>
            </div>
            <div style='text-align:right;margin-top:3px;'>
                <span style='font-size:0.62rem;color:#A89880;'>{_diff_label} · ε = {epsilon:.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔒 Lock", key="lock_tracker"):
            st.session_state.noise_tracker_unlocked = False
            st.rerun()
    else:
        st.markdown("""
        <div class='dp-notice' style='padding:12px 14px;text-align:center;'>
            <div style='font-size:1.4rem;margin-bottom:6px;'>🔒</div>
            <div style='font-size:0.72rem;color:#A89880;'>Restricted to authorised<br>warden personnel only</div>
        </div>
        """, unsafe_allow_html=True)
        _pin = st.text_input("Enter warden password", type="password", key="tracker_pin", label_visibility="collapsed", placeholder="Enter password")
        if _pin == "wardern1":
            st.session_state.noise_tracker_unlocked = True
            st.rerun()
        elif _pin:
            st.error("Incorrect password.")


# ── Scenario type ─────────────────────────────────────────────────────────────
is_fire     = selected_base in FIRE_SCENARIOS
fire_meta   = FIRE_SCENARIOS.get(selected_base, {})
fire_origin = fire_meta.get("origin", "")
fire_active = False
if is_fire:
    fire_active = any(
        e in ["Fire Emergency", "Smoke Spread Alert", "Evacuation Update"]
        for e in df_now["Event_Type"].unique()
    )

# Pre-compute alarm_count early — used by ticker before the KPI row
alarm_count = int(df_now["Fire_Alarm_Triggered"].sum())

# ── Dynamic CSS: emergency mode shifts sidebar to dark ────────────────────────
if fire_active:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1A1210 !important;
        border-right: 1.5px solid #5A2020 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #EDE5DC !important;
    }
    [data-testid="stSidebar"] hr { border-color: #5A2020 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: #842420 !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover { background: #6A1810 !important; }

    /* Page-edge red vignette */
    [data-testid="stAppViewContainer"] > section:first-child {
        box-shadow: inset 0 0 140px rgba(132,36,32,0.13) !important;
    }

    /* Ripple dot */
    @keyframes ripple {
        0%   { transform: scale(1);   opacity: 0.7; }
        100% { transform: scale(3.2); opacity: 0; }
    }
    .fire-dot-ring {
        position:absolute; top:0; left:0;
        width:12px; height:12px;
        background:#FF3030; border-radius:50%;
        animation: ripple 1.8s ease-out infinite;
    }
    .fire-dot-core {
        position:absolute; top:0; left:0;
        width:12px; height:12px;
        background:#FF4040; border-radius:50%;
    }

    /* Scrolling ticker */
    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* Alert banner — solid dark red in fire mode */
    .alert-banner-fire {
        background: #5A1410 !important;
        border-color: #842420 !important;
        border-left-color: #FF3030 !important;
    }
    .alert-banner-fire .alert-banner-text,
    .alert-banner-fire .alert-banner-bold { color: #FDECEA !important; }
    .alert-banner-fire .alert-banner-icon  { color: #FF7060 !important; }

    /* Zone card — fire origin pulsing border */
    @keyframes origin-flash {
        0%, 100% { border-left-color: #C84040; box-shadow: none; }
        50%       { border-left-color: #FF3030; box-shadow: 0 0 16px rgba(132,36,32,0.4); }
    }
    .zone-card-origin { animation: origin-flash 1.8s ease-in-out infinite; }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Overview", "Zone Details", "Incident Report"])

with tab1:

    # ── Emergency command strip (fire mode only) ──────────────────────────────────
    if fire_active:
        _fe_head = df_scenario[
            (df_scenario["Timestamp"] <= current_ts) &
            (df_scenario["Event_Type"] == "Fire Emergency")
        ]
        _fire_start_ts = _fe_head["Timestamp"].min() if not _fe_head.empty else current_ts
        _elapsed_min = int((current_ts - _fire_start_ts).total_seconds() / 60)
        _elapsed_str = f"{_elapsed_min // 60:02d}h {_elapsed_min % 60:02d}m"
        _origin_name_h = ZONE_META.get(fire_origin, fire_origin)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1A1210 0%,#2C1810 100%);
                    border-radius:10px;padding:14px 22px;margin-bottom:14px;
                    display:flex;justify-content:space-between;align-items:center;
                    border-left:5px solid #842420;'>
            <div>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:5px;'>
                    <div style='position:relative;width:12px;height:12px;flex-shrink:0;'>
                        <span class='fire-dot-ring'></span>
                        <span class='fire-dot-core'></span>
                    </div>
                    <span style='color:#EDD5D3;font-size:0.64rem;font-weight:700;
                                 letter-spacing:0.14em;text-transform:uppercase;
                                 font-family:"DM Sans",sans-serif;'>Fire Emergency Active</span>
                </div>
                <div style='color:#FFFFFF;font-size:1.05rem;font-weight:700;
                            font-family:"DM Sans",sans-serif;line-height:1.3;'>
                    Centre for Technology &amp; Infusion
                    &ensp;&middot;&ensp;{fire_meta.get("level","")}
                    &ensp;&middot;&ensp;Origin: <span style='color:#EDD5D3;'>{fire_origin} — {_origin_name_h}</span>
                </div>
            </div>
            <div style='text-align:right;flex-shrink:0;padding-left:20px;'>
                <div style='color:#A89880;font-size:0.62rem;letter-spacing:0.08em;
                            text-transform:uppercase;font-weight:600;
                            font-family:"DM Sans",sans-serif;'>Incident duration</div>
                <div style='font-family:"DM Mono",monospace;color:#FFFFFF;
                            font-size:1.55rem;font-weight:500;line-height:1.1;'>{_elapsed_str}</div>
                <div style='color:#A89880;font-size:0.68rem;margin-top:3px;
                            font-family:"DM Mono",monospace;'>{current_ts.strftime('%H:%M · %a %d %b')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Scrolling ticker
        _ticker_text = (
            f"FIRE EMERGENCY ACTIVE &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"ORIGIN: {fire_origin} — {_origin_name_h.upper()} &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"{alarm_count} ALARM{'S' if alarm_count != 1 else ''} TRIGGERED &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"ALL WARDENS RESPOND IMMEDIATELY &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"INITIATE EVACUATION PROTOCOL &nbsp;&nbsp;·&nbsp;&nbsp; "
        )
        st.markdown(f"""
        <div style='background:#842420;padding:7px 0;overflow:hidden;
                    border-radius:7px;margin-bottom:14px;margin-top:-6px;'>
            <div style='display:flex;animation:ticker-scroll 24s linear infinite;
                        white-space:nowrap;width:max-content;'>
                <span style='color:#FFFFFF;font-size:0.68rem;font-weight:700;
                             letter-spacing:0.1em;padding:0 50px;'>{_ticker_text}</span>
                <span style='color:#FFFFFF;font-size:0.68rem;font-weight:700;
                             letter-spacing:0.1em;padding:0 50px;'>{_ticker_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Page header ───────────────────────────────────────────────────────────────
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.markdown("<h1>Operations Centre</h1>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#A89880;font-size:0.85rem;margin-top:-10px;'>{selected_label}</p>",
            unsafe_allow_html=True,
        )

    with col_meta:
        next_refresh_secs = 30 - (refresh_count % 1)   # approximate countdown label
        live_status = "LIVE" if not st.session_state.paused else "PAUSED"
        live_color  = "#842420" if not st.session_state.paused else "#8A6E15"
        st.markdown(f"""
        <div style='text-align:right;padding-top:12px;'>
            <div style='font-family:DM Mono,monospace;font-size:1.1rem;font-weight:500;color:#2C1810;'>
                {current_ts.strftime('%H:%M')}
            </div>
            <div style='font-size:0.75rem;color:#A89880;'>{current_ts.strftime('%a %d %b %Y')}</div>
            <div style='font-size:0.7rem;color:#B0A090;margin-bottom:6px;'>
                Tick {st.session_state.tick + 1} / {n_ticks}
            </div>
            <span class='live-badge' style='background:{"#EDD5D3" if not st.session_state.paused else "#F5EDD0"};
                  border-color:{"#C8A0A0" if not st.session_state.paused else "#D4B840"};
                  color:{live_color};'>
                <span class='live-dot' style='background:{live_color};'></span>
                {live_status} · 30s
            </span>
        </div>
        """, unsafe_allow_html=True)


    # ── Alert banner ──────────────────────────────────────────────────────────────
    if is_fire and fire_active:
        origin_name  = ZONE_META.get(fire_origin, fire_origin)
        alarm_count  = int(df_now["Fire_Alarm_Triggered"].sum())
        st.markdown(f"""
        <div class='alert-banner alert-banner-fire'>
            <div class='alert-banner-icon'>{ICONS["alert"]}</div>
            <div class='alert-banner-text'>
                <span class='alert-banner-bold'>FIRE EMERGENCY ACTIVE</span> ·
                Origin: <span class='alert-banner-bold'>{fire_origin} — {origin_name}</span> ·
                {alarm_count} alarm{'s' if alarm_count != 1 else ''} triggered ·
                Wardens: initiate evacuation protocol
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif is_fire:
        st.markdown(
            f"""<div style='display:inline-flex;align-items:center;gap:7px;
                background:#F5EDD0;border:1px solid #C8A830;border-radius:20px;
                padding:4px 12px;margin-bottom:10px;'>
                <span style='font-size:0.78rem;'>{ICONS['clock']}</span>
                <span style='font-size:0.75rem;font-weight:600;color:#7A5A10;
                             letter-spacing:0.03em;'>Pre-incident monitoring
                    <span style='font-weight:400;color:#A08020;margin-left:4px;'>
                        — fire not yet started
                    </span>
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style='display:inline-flex;align-items:center;gap:7px;
                background:#E5ECDE;border:1px solid #AACBA0;border-radius:20px;
                padding:4px 12px;margin-bottom:10px;'>
                <span style='font-size:0.78rem;'>{ICONS['check']}</span>
                <span style='font-size:0.75rem;font-weight:600;color:#3A5A2A;
                             letter-spacing:0.03em;'>Routine monitoring
                    <span style='font-weight:400;color:#6A8A5A;margin-left:4px;'>
                        — all zones normal
                    </span>
                </span>
            </div>""",
            unsafe_allow_html=True,
        )


    # ── Computed KPI values ───────────────────────────────────────────────────────
    total_occ        = int(df_now["DP_Total_Occupancy"].sum())
    total_vulnerable = int(df_now["DP_Vulnerable_Count"].sum())
    total_children   = int(df_now["DP_Child_Count"].sum())
    total_elderly    = int(df_now["DP_Elderly_Count"].sum())
    total_disabled   = int(df_now["DP_Disabled_Count"].sum())
    total_adults     = int(df_now["DP_Adult_Count"].sum())
    alarm_count      = int(df_now["Fire_Alarm_Triggered"].sum())
    cleared_count    = int((df_now["Zone_Clearance_Status"] == "Cleared").sum())
    total_zones      = len(df_now)
    critical_zones   = int(df_now["Hazard_Level_Rating"].isin(["High", "Critical"]).sum())

    # ── Event data (used in both col_right current feed and Incident Report) ────────
    df_history = df_scenario[df_scenario["Timestamp"] <= current_ts].copy()
    EVENT_META = {
        "Fire Emergency":    {"icon": ICONS["fire"],  "label": "FIRE",       "color": "#842420", "badge_bg": "#EDD5D3"},
        "Smoke Spread Alert":{"icon": ICONS["wind"],  "label": "SMOKE",      "color": "#B01808", "badge_bg": "#F5DED8"},
        "Evacuation Update": {"icon": ICONS["evac"],  "label": "EVACUATION", "color": "#8A6E15", "badge_bg": "#F5EDD0"},
        "Routine Monitoring":{"icon": ICONS["check"], "label": "ROUTINE",    "color": "#7A6A5A", "badge_bg": "#EDE5DC"},
    }

    # ── Main two-column layout: left = KPIs + zones, right = floor plan ──────────
    if st.session_state.heatmap_fullscreen:
        col_left  = None
        col_right = st.container()
    else:
        col_left, col_right = st.columns([2, 3], gap="medium")

    if not st.session_state.heatmap_fullscreen:
        with col_left:
            # ── Building Summary ─────────────────────────────
            st.markdown("<div class='section-label'>Building Summary</div>", unsafe_allow_html=True)
            kpi_data = [
                ("Total Occupancy (DP)", str(total_occ), "DP-protected headcount", "#465030"),
                ("Vulnerable Persons",   str(total_vulnerable), f"Children {total_children} · Elderly {total_elderly} · Disabled {total_disabled}", "#B01808"),
                ("Active Alarms",        str(alarm_count), "Zones with triggered alarms", "#842420" if alarm_count > 0 else "#465030"),
                ("Zones Cleared",        f"{cleared_count}/{total_zones}", "Warden-verified clearances", "#465030"),
                ("High-Risk Zones",      str(critical_zones), "Hazard level High or Critical", "#842420" if critical_zones > 0 else "#465030"),
            ]
            _kcols = st.columns(2)
            for i, (label, value, sub, accent) in enumerate(kpi_data):
                is_crit = fire_active and (
                    (label == "Active Alarms" and alarm_count > 0) or
                    (label == "High-Risk Zones" and critical_zones > 0)
                )
                extra_cls = " kpi-card-critical" if is_crit else ""
                with _kcols[i % 2]:
                    st.markdown(f"""
                    <div class='kpi-card{extra_cls}' style='--accent:{accent};margin-bottom:8px;'>
                        <div class='kpi-label'>{label}</div>
                        <div class='kpi-value'>{value}</div>
                        <div class='kpi-sub'>{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Adaptive ε warning ───────────────────────────
            _dp_mae     = float((df_now["DP_Total_Occupancy"] - df_now["Total_Occupancy_Raw"]).abs().mean())
            _avg_occ    = float(df_now["Total_Occupancy_Raw"].mean())
            _noise_pct  = (_dp_mae / max(_avg_occ, 1)) * 100

            if _dp_mae > 5.0 and _noise_pct > 15:
                st.markdown(f"""
                <div style='background:#F5EDD0;border:1.5px solid #D4B840;border-left:5px solid #8A6E15;
                            border-radius:8px;padding:12px 18px;margin-top:12px;'>
                    <span style='font-weight:700;color:#8A6E15;font-size:0.88rem;display:inline-flex;align-items:center;gap:6px;'>
                        {ICONS["warning"]} High Noise Warning
                    </span>
                    <span style='color:#6A5010;font-size:0.85rem;margin-left:8px;'>
                        DP noise averaging <b>{_dp_mae:.1f} people/zone</b> ({_noise_pct:.0f}% of mean occupancy).
                        Occupancy counts may be unreliable for operational decisions.
                    </span>
                    <div style='font-size:0.75rem;color:#8A6E15;margin-top:5px;'>
                        Increase ε in the sidebar to reduce noise · Current ε = {epsilon:.1f}
                        (Laplace scale = {1/epsilon:.2f})
                    </div>
                    <div style='font-size:0.68rem;color:#A08820;margin-top:4px;font-style:italic;'>
                        Research note: this warning is itself a privacy side-channel — its presence reveals
                        that ε is low, which could leak information to an observer about the privacy setting in use.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Zone Status ──────────────────────────────────
            st.markdown("<div class='section-label' style='margin-top:10px;'>Zone Status</div>", unsafe_allow_html=True)
            zone_cards_html = """
            <style>
              body { margin:0; font-family:'DM Sans',sans-serif; background:transparent; }
              .zone-card { background:#fff; border:1.5px solid #D0BCAD; border-radius:10px;
                           padding:14px 16px; margin-bottom:10px; }
              .zone-card-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px; }
              .zone-id   { font-size:0.82rem; font-weight:700; color:#465030; font-family:monospace;
                           background:#E5ECDE; padding:1px 7px; border-radius:5px; display:inline-block; }
              .zone-name { font-size:0.95rem; font-weight:600; color:#2C1810; }
              .zone-occ  { font-size:1.5rem; font-weight:500; color:#2C1810; text-align:right; font-family:monospace; }
              .zone-occ-label { font-size:0.68rem; color:#A89880; text-transform:uppercase; letter-spacing:0.05em; text-align:right; }
              .zone-pills { display:flex; flex-wrap:wrap; gap:5px; }
              .pill { font-size:0.68rem; font-weight:600; padding:2px 8px; border-radius:20px;
                      letter-spacing:0.04em; text-transform:uppercase; }
              .pill-Low      { background:#E5ECDE; color:#465030; }
              .pill-Moderate { background:#F5EDD0; color:#8A6E15; }
              .pill-High     { background:#F5DED8; color:#B01808; }
              .pill-Critical { background:#EDD5D3; color:#842420; }
              .pill-Cleared  { background:#E5ECDE; color:#465030; }
              .pill-Nearly   { background:#F5EDD0; color:#8A6E15; }
              .pill-Occupied { background:#F5DED8; color:#B01808; }
              .pill-NotStarted { background:#EDE5DC; color:#7A6A5A; }
              .pill-Yes      { background:#E5ECDE; color:#465030; }
              .pill-Partial  { background:#F5EDD0; color:#8A6E15; }
              .pill-No       { background:#EDD5D3; color:#842420; }
              .pill-alarm    { background:#EDD5D3; color:#842420; }
              .pill-sprinkler{ background:#DDE8F0; color:#2A5A80; }
              @keyframes origin-flash {
                0%, 100% { border-left-color: #C84040; box-shadow: none; }
                50%       { border-left-color: #FF3030; box-shadow: 0 0 16px rgba(132,36,32,0.4); }
              }
              .zone-card-origin { animation: origin-flash 1.8s ease-in-out infinite; }
            </style>
            """
            for _, row in df_now.sort_values("Zone_ID").iterrows():
                zone_id   = row["Zone_ID"]
                zone_name = ZONE_META.get(zone_id, str(row.get("Zone_Name", zone_id)))
                dp_occ    = int(row["DP_Total_Occupancy"])
                hazard    = str(row.get("Predicted_Hazard_Level", row.get("Hazard_Level_Rating", "Low")))
                clearance = str(row["Zone_Clearance_Status"])
                route     = str(row["Exit_Route_Passable"])
                alarm     = int(row["Fire_Alarm_Triggered"])
                sprinkler = str(row["Sprinkler_System_Status"])
                is_origin = zone_id == fire_origin and is_fire and fire_active

                clearance_pill  = {"Cleared": "pill-Cleared", "Nearly Cleared": "pill-Nearly",
                                   "Occupied": "pill-Occupied", "Not Started": "pill-NotStarted"}.get(clearance, "pill-NotStarted")
                clearance_short = {"Cleared": "Cleared", "Nearly Cleared": "~Cleared",
                                   "Occupied": "Occupied", "Not Started": "Monitoring"}.get(clearance, clearance)
                route_pill  = {"Yes": "pill-Yes", "Partial": "pill-Partial", "No": "pill-No"}.get(route, "pill-No")
                border       = "border-left:4px solid #842420;background:#EDD5D3;" if is_origin else ""
                origin_class = " zone-card-origin" if is_origin else ""
                alarm_pill  = f'<span class="pill pill-alarm">{ICONS["bell"]} Alarm</span>' if alarm else ""
                spkl_pill   = f'<span class="pill pill-sprinkler">{ICONS["droplet"]} Sprinkler Active</span>' if sprinkler == "Activated" else ""
                origin_lbl  = f' {ICONS["fire"]} FIRE ORIGIN' if is_origin else ""

                zone_cards_html += f"""
                <div class="zone-card{origin_class}" style="{border}">
                  <div class="zone-card-header">
                    <span>
                      <div class="zone-id">{zone_id}{origin_lbl}</div>
                      <div class="zone-name">{zone_name}</div>
                    </span>
                    <span>
                      <div class="zone-occ">{dp_occ}</div>
                      <div class="zone-occ-label">occupants (DP)</div>
                    </span>
                  </div>
                  <div class="zone-pills">
                    <span class="pill pill-{hazard}">{hazard} Hazard</span>
                    <span class="pill {clearance_pill}">{clearance_short}</span>
                    <span class="pill {route_pill}">Exit: {route}</span>
                    {alarm_pill}{spkl_pill}
                  </div>
                </div>"""

            components.html(zone_cards_html, height=len(df_now) * 120, scrolling=True)

    with col_right:
        # Header row: title left, expand button right
        _rh_title, _rh_btn = st.columns([4, 1])
        with _rh_title:
            st.markdown("**Floor Plan — Hazard & Occupancy**")
            st.markdown("<p style='font-size:0.78rem;color:#A89880;margin-top:-8px;'>Bubble size = occupancy · colour = vulnerability · gradient = crowd density (DP-protected)</p>", unsafe_allow_html=True)
        with _rh_btn:
            _fs_label = "⛶  Minimise" if st.session_state.heatmap_fullscreen else "⛶  Expand"
            if st.button(_fs_label, key="heatmap_toggle"):
                st.session_state.heatmap_fullscreen = not st.session_state.heatmap_fullscreen
                st.rerun()

        HAZARD_FILL = {
            "Low":      "rgba(70,80,48,0.15)",
            "Moderate": "rgba(192,164,37,0.20)",
            "High":     "rgba(176,24,8,0.25)",
            "Critical": "rgba(132,36,32,0.35)",
        }
        HAZARD_LINE = {"Low": "#6AE05A", "Moderate": "#F0D060", "High": "#FF6040", "Critical": "#FF2020"}

        fig_map = go.Figure()

        # ── Floor map image as background ────────────────────────────────────
        if FLOOR_MAP_URI:
            fig_map.add_layout_image(dict(
                source=FLOOR_MAP_URI,
                xref="x", yref="y",
                x=0, y=0,
                sizex=100, sizey=100,
                xanchor="left", yanchor="top",
                sizing="stretch",
                opacity=map_opacity,
                layer="below",
            ))

        # ── LAYER 1: Occupancy density (teal/blue, always on) ────────────────
        if not df_heatmap.empty:
            fig_map.add_trace(go.Histogram2dContour(
                x=df_heatmap["DP_Coord_X"],
                y=df_heatmap["DP_Coord_Y"],
                z=df_heatmap["DP_Cluster_Size"],
                colorscale=[
                    [0.0, "rgba(30,80,160,0)"],
                    [0.3, "rgba(30,120,200,0.30)"],
                    [0.6, "rgba(20,160,180,0.50)"],
                    [1.0, "rgba(10,200,160,0.70)"],
                ],
                reversescale=False,
                opacity=0.55,
                histfunc="sum",
                ncontours=12,
                showscale=False,
                line=dict(width=0),
                hoverinfo="skip",
                name="Occupancy density",
            ))

        # ── LAYER 2: Fire hazard intensity (red/orange, only when fire active) ─
        # Builds a Gaussian point cloud per zone weighted by Predicted_Hazard_Score
        # so the gradient peaks at dangerous zones and fades toward safe ones.
        if is_fire and fire_active and not df_now.empty:
            _rng = np.random.default_rng(99)
            _hx, _hy, _hz = [], [], []
            for _, _zrow in df_now.iterrows():
                _hscore = float(_zrow.get("Predicted_Hazard_Score", 0))
                if _hscore < 5:
                    continue
                _cx = float(_zrow["Zone_Center_X"])
                _cy = float(_zrow["Zone_Center_Y"])
                _sx = (float(_zrow["Zone_X_Max"]) - float(_zrow["Zone_X_Min"])) / 4
                _sy = (float(_zrow["Zone_Y_Max"]) - float(_zrow["Zone_Y_Min"])) / 4
                _n  = max(15, int(_hscore * 3))
                _hx.extend(_rng.normal(_cx, _sx, _n).tolist())
                _hy.extend(_rng.normal(_cy, _sy, _n).tolist())
                _hz.extend([_hscore] * _n)
            if _hx:
                fig_map.add_trace(go.Histogram2dContour(
                    x=_hx, y=_hy, z=_hz,
                    colorscale="Hot",
                    reversescale=False,
                    opacity=heat_intensity,
                    histfunc="sum",
                    ncontours=15,
                    showscale=False,
                    line=dict(width=0),
                    hoverinfo="skip",
                    name="Fire hazard intensity",
                ))

        # ── Zone hazard tint rectangles ───────────────────────────────────────
        for _, row in df_now.iterrows():
            zid = row["Zone_ID"]
            haz = str(row["Predicted_Hazard_Level"])
            fig_map.add_shape(
                type="rect",
                x0=float(row["Zone_X_Min"]), y0=float(row["Zone_Y_Min"]),
                x1=float(row["Zone_X_Max"]), y1=float(row["Zone_Y_Max"]),
                fillcolor=HAZARD_FILL.get(haz, "rgba(70,80,48,0.15)"),
                line=dict(color=HAZARD_LINE.get(haz, "#6AE05A"), width=2),
                layer="below",
            )

        # ── Occupancy bubbles ─────────────────────────────────────────────────
        global_max_occ = max(int(df_now["DP_Total_Occupancy"].max()), 1)

        for _, row in df_now.sort_values("Zone_ID").iterrows():
            zid         = row["Zone_ID"]
            cx          = float(row["Zone_Center_X"])
            cy          = float(row["Zone_Center_Y"])
            dp_occ      = max(0, int(row["DP_Total_Occupancy"]))
            dp_vuln     = max(0, int(row["DP_Vulnerable_Count"]))
            child_cnt   = max(0, int(row["DP_Child_Count"]))
            elderly_cnt = max(0, int(row["DP_Elderly_Count"]))
            disabled_cnt= max(0, int(row["DP_Disabled_Count"]))
            is_vuln     = dp_vuln > 0
            name        = ZONE_META.get(zid, zid)
            haz         = str(row["Predicted_Hazard_Level"])

            # Proportional size: 14–42px base, scaled by bubble_scale slider
            base_px  = 14 + (42 - 14) * (max(1, dp_occ) / global_max_occ)
            px_size  = int(base_px * bubble_scale)

            # Shadow disc — punches through heatmap
            fig_map.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(size=px_size + 8, sizemode="diameter",
                            color="rgba(0,0,0,0.40)", line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ))

            # Colour: amber = vulnerable zones, blue = normal
            if is_vuln:
                fill = "rgba(210,100,20,0.78)"
                rim  = "rgba(255,160,60,0.95)"
                ring = "rgba(255,140,0,0.70)"
            else:
                fill = "rgba(40,130,180,0.72)"
                rim  = "rgba(100,190,210,0.85)"
                ring = "rgba(80,200,220,0.6)"

            # Glow ring
            fig_map.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(size=px_size + 5, sizemode="diameter",
                            color="rgba(0,0,0,0)", line=dict(width=3, color=ring)),
                hoverinfo="skip", showlegend=False,
            ))

            # Hover breakdown for vulnerable
            vuln_lines = ""
            if is_vuln:
                vuln_lines = f"<br><b>⚠ Vulnerable ({dp_vuln} total)</b>"
                if child_cnt    > 0: vuln_lines += f"<br> Children: {child_cnt}"
                if elderly_cnt  > 0: vuln_lines += f"<br> Elderly: {elderly_cnt}"
                if disabled_cnt > 0: vuln_lines += f"<br> Disabled: {disabled_cnt}"

            # Main bubble with count label
            fig_map.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(size=px_size, sizemode="diameter", color=fill,
                            line=dict(width=3 if is_vuln else 2, color=rim)),
                text=[f"~{dp_occ}"],
                textfont=dict(color="white", size=max(8, int(px_size * 0.32)), family="DM Mono"),
                textposition="middle center",
                hovertemplate=(
                    f"<b>{zid} — {name}</b><br>"
                    f"Occupancy: ~{dp_occ} (DP-protected)<br>"
                    f"Hazard: {haz}"
                    f"{vuln_lines}<extra></extra>"
                ),
                showlegend=False,
            ))

            # ⚠ Warning badge for vulnerable zones
            if is_vuln:
                badge_off = (px_size / 2 + 4) / 10
                badge_px  = max(14, int(px_size * 0.45))
                fig_map.add_trace(go.Scatter(
                    x=[cx + badge_off], y=[cy + badge_off],
                    mode="markers+text",
                    marker=dict(size=badge_px, sizemode="diameter",
                                color="rgba(180,40,0,0.90)",
                                line=dict(width=2, color="rgba(255,180,60,1.0)")),
                    text=["⚠"],
                    textfont=dict(color="white", size=max(7, int(badge_px * 0.45))),
                    textposition="middle center",
                    hovertemplate=(
                        f"<b>⚠ Vulnerable — {zid}</b><br>"
                        + (f"Children: {child_cnt}<br>"   if child_cnt    > 0 else "")
                        + (f"Elderly: {elderly_cnt}<br>"  if elderly_cnt  > 0 else "")
                        + (f"Disabled: {disabled_cnt}<br>"if disabled_cnt > 0 else "")
                        + f"<b>Total at-risk: {dp_vuln}</b><extra></extra>"
                    ),
                    showlegend=False,
                ))

        # Fire origin — blinking red alert light at top-right corner of zone
        if is_fire and fire_active:
            origin_row = df_now[df_now["Zone_ID"] == fire_origin]
            if not origin_row.empty:
                # top-right corner: X_Max, Y_Min (y-axis is reversed so Y_Min = visual top)
                corner_x = float(origin_row["Zone_X_Max"].iloc[0]) - 1.5
                corner_y = float(origin_row["Zone_Y_Min"].iloc[0]) + 1.5
                # Outer glow ring (large, semi-transparent)
                fig_map.add_trace(go.Scatter(
                    x=[corner_x], y=[corner_y],
                    mode="markers",
                    marker=dict(size=22, color="rgba(255,30,30,0.25)", line=dict(width=0)),
                    hoverinfo="skip", showlegend=False,
                ))
                # Inner solid red dot
                fig_map.add_trace(go.Scatter(
                    x=[corner_x], y=[corner_y],
                    mode="markers",
                    marker=dict(size=11, color="#FF1E1E", line=dict(width=2, color="#FF9090")),
                    hoverinfo="skip", showlegend=False,
                ))

        fig_map.update_layout(
            height=720 if st.session_state.heatmap_fullscreen else 520,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
            showlegend=False,
            dragmode=False,
            font=dict(family="DM Sans", color="#2C1810"),
            annotations=[],
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        st.markdown("""
        <div style='display:flex;align-items:center;flex-wrap:wrap;gap:12px;
                    padding:6px 14px;margin-top:-12px;margin-bottom:6px;
                    background:#F5F0EB;border:1px solid #D0BCAD;border-radius:8px;
                    font-size:0.71rem;color:#4A3A2A;'>
            <span style='font-size:0.6rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;
                         color:#A89880;padding-right:12px;border-right:1px solid #D0BCAD;white-space:nowrap;'>Map Key</span>
            <span style='display:inline-flex;align-items:center;gap:5px;'>
                <span style='width:10px;height:10px;border-radius:50%;flex-shrink:0;
                             background:rgba(40,130,180,0.78);border:1.5px solid rgba(100,190,210,0.9);'></span>Occupancy (DP)</span>
            <span style='display:inline-flex;align-items:center;gap:5px;'>
                <span style='width:10px;height:10px;border-radius:50%;flex-shrink:0;
                             background:rgba(210,100,20,0.78);border:1.5px solid rgba(255,160,60,0.9);'></span>Vulnerable Zone</span>
            <span style='display:inline-flex;align-items:center;gap:5px;'>
                <span style='width:14px;height:6px;border-radius:3px;flex-shrink:0;
                             background:linear-gradient(90deg,rgba(30,120,200,0.3),rgba(10,200,160,0.65));'></span>Crowd Density</span>
            <span style='display:inline-flex;align-items:center;gap:5px;'>
                <span style='width:14px;height:6px;border-radius:3px;flex-shrink:0;
                             background:linear-gradient(90deg,rgba(180,0,0,0.7),rgba(255,180,0,0.6));'></span>Fire Hazard (ML)</span>
            <span style='display:inline-flex;align-items:center;gap:5px;'>
                <span style='width:10px;height:10px;border-radius:2px;flex-shrink:0;
                             background:rgba(176,24,8,0.2);border:1.5px solid #FF6040;'></span>High Hazard (&gt;70%)</span>
            <span style='display:inline-flex;align-items:center;gap:5px;'>
                <span style='width:10px;height:10px;border-radius:2px;flex-shrink:0;
                             background:rgba(192,164,37,0.2);border:1.5px solid #F0D060;'></span>Moderate (30–70%)</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='dp-notice'>
            Occupancy counts are DP-protected (ε={epsilon:.1f}, Laplace). Dot positions are randomly
            simulated within each zone — no real individual locations are tracked or stored.
        </div>
        """, unsafe_allow_html=True)

        # ── Current events feed (right column, below floor plan) ─────────────────
        st.markdown(
            "<div class='section-label' style='margin-top:10px;'>Current Events</div>",
            unsafe_allow_html=True,
        )
        _df_now_events = (
            df_history[df_history["Timestamp"] == current_ts][
                ["Timestamp", "Zone_ID", "Event_Type", "Details"]
            ]
            .drop_duplicates(subset=["Zone_ID", "Event_Type"])
            .sort_values("Zone_ID")
        )
        _feed_now_html = """<style>
          body{margin:0;font-family:'DM Sans',sans-serif;background:transparent;}
          .cf{background:#fff;border:1.5px solid #D0BCAD;border-radius:10px;overflow:hidden;}
          .cr{display:flex;align-items:flex-start;gap:8px;padding:9px 12px;
              border-bottom:1px solid #F5F3EE;border-left:4px solid transparent;}
          .cr:last-child{border-bottom:none;}
          .ct{font-family:monospace;font-size:0.72rem;color:#B0A090;white-space:nowrap;padding-top:2px;}
          .ci{font-size:0.95rem;flex-shrink:0;padding-top:1px;line-height:1;}
          .cb{font-size:0.58rem;font-weight:700;padding:2px 6px;border-radius:20px;
              letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;}
          .cz{font-family:monospace;font-size:0.72rem;font-weight:700;color:#465030;}
          .cd{font-size:0.68rem;color:#7A6A5A;margin-top:2px;line-height:1.4;}
        </style><div class='cf'>"""
        if _df_now_events.empty:
            _feed_now_html += "<div class='cr'><span class='cd' style='color:#A89880;'>No events at this timestamp.</span></div>"
        else:
            for _, _r in _df_now_events.iterrows():
                _m   = EVENT_META.get(str(_r["Event_Type"]), EVENT_META["Routine Monitoring"])
                _zid = str(_r["Zone_ID"])
                _det = str(_r["Details"])
                if len(_det) > 90: _det = _det[:90] + "…"
                _feed_now_html += f"""
                <div class='cr' style='border-left-color:{_m["color"]};'>
                  <span class='ci'>{_m["icon"]}</span>
                  <span style='flex:1;min-width:0;'>
                    <div style='display:flex;align-items:center;gap:5px;flex-wrap:wrap;'>
                      <span class='cb' style='background:{_m["badge_bg"]};color:{_m["color"]};'>{_m["label"]}</span>
                      <span class='cz'>{_zid}</span>
                      <span style='font-size:0.68rem;color:#A89880;'>· {ZONE_META.get(_zid,_zid)}</span>
                    </div>
                    <div class='cd'>{_det}</div>
                  </span>
                </div>"""
        _feed_now_html += "</div>"
        _n_now = max(len(_df_now_events), 1)
        components.html(_feed_now_html, height=min(_n_now * 72 + 16, 340), scrolling=True)



    # ── DP Explainer ──────────────────────────────────────────────────────────────
    with st.expander("Privacy Budget Explainer — how does ε work?", expanded=False):
        st.markdown(
            "Adjust the **ε slider** in the sidebar and watch all three charts update live.",
            unsafe_allow_html=False,
        )
        dp_col1, dp_col2, dp_col3 = st.columns(3)

        # ── 1. Laplace noise distribution curve ───────────────────────────────────
        with dp_col1:
            st.markdown("**Noise Distribution**")
            st.markdown(
                "<p style='font-size:0.75rem;color:#A89880;margin-top:-8px;'>"
                "Shape of noise added to each count</p>",
                unsafe_allow_html=True,
            )
            x_noise = np.linspace(-10, 10, 400)
            fig_lap  = go.Figure()
            ref_eps  = {"ε=0.5 (strong)": 0.5, "ε=2.0 (weak)": 2.0}
            ref_cols = {"ε=0.5 (strong)": "#C08A80", "ε=2.0 (weak)": "#C0A425"}
            for label, eps_ref in ref_eps.items():
                scale_ref = 1.0 / eps_ref
                y_ref = (1 / (2 * scale_ref)) * np.exp(-np.abs(x_noise) / scale_ref)
                fig_lap.add_trace(go.Scatter(
                    x=x_noise, y=y_ref, mode="lines", name=label,
                    line=dict(color=ref_cols[label], width=1.5, dash="dot"),
                ))
            scale_cur = 1.0 / epsilon
            y_cur = (1 / (2 * scale_cur)) * np.exp(-np.abs(x_noise) / scale_cur)
            fig_lap.add_trace(go.Scatter(
                x=x_noise, y=y_cur, mode="lines",
                name=f"ε={epsilon:.1f} (current)",
                line=dict(color="#842420", width=2.5),
                fill="tozeroy", fillcolor="rgba(132,36,32,0.08)",
            ))
            fig_lap.update_layout(
                height=220, margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                xaxis=dict(title="Noise added", tickfont=dict(size=8), showgrid=False,
                           zeroline=True, zerolinecolor="#D0BCAD"),
                yaxis=dict(title="Probability", tickfont=dict(size=8), showgrid=True,
                           gridcolor="#EDE5DC"),
                legend=dict(font=dict(size=8), x=0, y=1),
                font=dict(family="DM Sans", size=9),
            )
            st.plotly_chart(fig_lap, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<p style='font-size:0.72rem;color:#A89880;'>"
                f"Current scale = <b>{scale_cur:.2f}</b>. "
                f"{'Tall & narrow = less noise = weaker privacy.' if epsilon > 2 else 'Wide & flat = more noise = stronger privacy.'}</p>",
                unsafe_allow_html=True,
            )

        # ── 2. True vs DP per zone ────────────────────────────────────────────────
        with dp_col2:
            st.markdown("**True vs DP Count (current tick)**")
            st.markdown(
                "<p style='font-size:0.75rem;color:#A89880;margin-top:-8px;'>"
                "Gap between bars = noise introduced</p>",
                unsafe_allow_html=True,
            )
            zones_sorted = df_now.sort_values("Zone_ID")
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=zones_sorted["Zone_ID"],
                y=zones_sorted["Total_Occupancy_Raw"],
                name="True count",
                marker=dict(color="#465030", opacity=0.3),
                hovertemplate="%{x}: %{y} (true)<extra></extra>",
            ))
            fig_cmp.add_trace(go.Bar(
                x=zones_sorted["Zone_ID"],
                y=zones_sorted["DP_Total_Occupancy"],
                name="DP count",
                marker=dict(color="#842420", opacity=0.8),
                hovertemplate="%{x}: %{y} (DP)<extra></extra>",
            ))
            fig_cmp.update_layout(
                height=220, margin=dict(l=0, r=0, t=5, b=0),
                barmode="overlay",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                xaxis=dict(tickfont=dict(size=9), showgrid=False),
                yaxis=dict(tickfont=dict(size=9), showgrid=True, gridcolor="#EDE5DC",
                           title="Occupancy"),
                legend=dict(font=dict(size=8), x=0, y=1),
                font=dict(family="DM Sans", size=9),
            )
            st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
            mae_now = float((zones_sorted["DP_Total_Occupancy"] - zones_sorted["Total_Occupancy_Raw"]).abs().mean())
            st.markdown(
                f"<p style='font-size:0.72rem;color:#A89880;'>"
                f"Mean absolute error at ε={epsilon:.1f}: <b>{mae_now:.1f} people</b> per zone</p>",
                unsafe_allow_html=True,
            )

        # ── 3. ε vs accuracy trade-off ────────────────────────────────────────────
        with dp_col3:
            st.markdown("**Privacy–Utility Trade-off**")
            st.markdown(
                "<p style='font-size:0.75rem;color:#A89880;margin-top:-8px;'>"
                "How accuracy changes with ε</p>",
                unsafe_allow_html=True,
            )
            eps_range = np.round(np.arange(0.1, 5.1, 0.2), 1)
            mae_vals  = []
            raw       = df_now["Total_Occupancy_Raw"].values.astype(float)
            rng_tradeoff = np.random.default_rng(seed=st.session_state.tick * 999)
            for eps_val in eps_range:
                noise = rng_tradeoff.laplace(0, 1.0 / eps_val, (50, len(raw)))
                noisy = np.clip(raw + noise, 0, None).round()
                mae_vals.append(float(np.abs(noisy - raw).mean()))

            fig_tradeoff = go.Figure()
            fig_tradeoff.add_trace(go.Scatter(
                x=eps_range, y=mae_vals, mode="lines+markers",
                line=dict(color="#465030", width=2),
                marker=dict(size=4, color="#465030"),
                fill="tozeroy", fillcolor="rgba(70,80,48,0.07)",
                hovertemplate="ε=%{x:.1f} → MAE=%{y:.2f}<extra></extra>",
            ))
            # Mark current epsilon
            fig_tradeoff.add_shape(
                type="line", xref="x", yref="paper",
                x0=epsilon, x1=epsilon, y0=0, y1=1,
                line=dict(color="#842420", dash="dot", width=1.5),
            )
            fig_tradeoff.add_annotation(
                x=epsilon, y=1, xref="x", yref="paper",
                text=f"ε={epsilon:.1f}", showarrow=False,
                font=dict(size=9, color="#842420"), yanchor="bottom",
            )
            fig_tradeoff.update_layout(
                height=220, margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                xaxis=dict(title="ε (privacy budget)", tickfont=dict(size=8),
                           showgrid=False),
                yaxis=dict(title="Mean Abs. Error", tickfont=dict(size=8),
                           showgrid=True, gridcolor="#EDE5DC"),
                font=dict(family="DM Sans", size=9),
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                "<p style='font-size:0.72rem;color:#A89880;'>"
                "Lower ε → higher error → stronger privacy. "
                "Higher ε → lower error → weaker privacy.</p>",
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-top:30px;padding-top:14px;border-top:1.5px solid #D0BCAD;
                font-size:0.72rem;color:#CCCCCC;display:flex;justify-content:space-between;'>
        <span>Privacy-First AI for Emergency Response · Centre for Technology and Infusion, La Trobe University</span>
        <span>All occupancy data is differentially private. No personal identifiers are stored or displayed.</span>
    </div>
    """, unsafe_allow_html=True)
with tab2:
    st.markdown("<h2 style='margin-top:1rem;'>Zone Details</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#A89880;font-size:0.85rem;margin-top:-10px;'>Drill-down sensor and evacuation data for the selected zone</p>",
        unsafe_allow_html=True,
    )
    st.info("Zone Details page — coming soon.")

with tab3:
    st.markdown("<h2 style='margin-top:1rem;'>Incident Report</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#A89880;font-size:0.85rem;margin-top:-10px;'>Post-incident summary with downloadable CSV</p>",
        unsafe_allow_html=True,
    )
    st.info("Incident Report page — coming soon.")
