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
import streamlit.components.v1 as components


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
/* When sidebar is collapsed, keep a narrow strip visible with just the toggle button */
section[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    min-width: 2.5rem !important;
    width: 2.5rem !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] {
    display: none !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] {
    display: flex !important;
    justify-content: center !important;
    padding: 0.5rem 0 !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] button {
    display: flex !important;
    background: #465030 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stSidebarHeader"] button {
    background: #465030 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    opacity: 1 !important;
}
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
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FAF7F4 !important;
    border-color: #842420 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: #FF3030 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] p,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"] {
    color: #3D1A14 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #842420 !important; }
[data-testid="stSidebar"] [data-baseweb="popover"] [data-baseweb="menu"],
[data-testid="stSidebar"] ul[data-baseweb="menu"] {
    background-color: #F5E8E6 !important;
    border: 1.5px solid #C8A0A0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="menu"] li,
[data-testid="stSidebar"] [data-baseweb="menu"] [role="option"] {
    background-color: #F5E8E6 !important;
    color: #3D1A14 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-baseweb="menu"] li:hover,
[data-testid="stSidebar"] [data-baseweb="menu"] [aria-selected="true"],
[data-testid="stSidebar"] [data-baseweb="menu"] [role="option"]:hover {
    background-color: #EDD5D3 !important;
    color: #3D1A14 !important;
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
    selected_label = st.selectbox("Active Session", list(scenario_options.keys()))
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
        if _pin == "warden1":
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
        background-color: #3D1A14 !important;
        border-right: 1.5px solid #7A3030 !important;
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

# ── Helper functions (used by Zone Details tab) ───────────────────────────────
def infer_zone_status(row) -> str:
    haz   = str(row.get("Predicted_Hazard_Level", row.get("Hazard_Level_Rating", "")))
    alarm = int(row.get("Fire_Alarm_Triggered", 0))
    if alarm and haz in ("High", "Critical"):
        return "FIRE EMERGENCY"
    if haz == "Critical":
        return "CRITICAL HAZARD"
    if haz == "High":
        return "HIGH HAZARD"
    if haz == "Moderate":
        return "ELEVATED"
    return "MONITORING"


def route_label(val: str) -> str:
    return {"Yes": "Route Clear", "Partial": "Partial Block", "No": "Route Blocked"}.get(str(val), str(val))


def normalize(value, low: float, high: float) -> float:
    try:
        return max(0.0, min((float(value) - low) / max(high - low, 1e-9), 1.0))
    except (TypeError, ValueError):
        return 0.0


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
        <div style='background:linear-gradient(135deg,#3D1A14 0%,#4A1B15 100%);
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
    st.markdown("""
    <style>
    .zd-hero {
        background: linear-gradient(135deg, #2C1810 0%, #4A1B15 100%);
        border-left: 5px solid #842420;
        border-radius: 10px;
        padding: 14px 18px;
        color: #FFFFFF;
        margin-top: 1rem;
        margin-bottom: 14px;
        box-shadow: 0 10px 22px rgba(44,24,16,0.08);
    }
    .zd-hero-title { font-size: 1.65rem; font-weight: 700; color: #FFFFFF; }
    .zd-hero-sub { font-size: 0.9rem; color: #EDD5D3; margin-top: 4px; }
    .zd-status {
        display:inline-block; padding: 5px 12px; border-radius: 999px; font-size: 0.72rem;
        font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        background: #EDD5D3; color: #842420;
    }
    .zd-context {
        background: #F5F0EB; border: 1.5px solid #D0BCAD; border-radius: 10px;
        padding: 12px 16px; color: #2C1810; margin-bottom: 14px;
        font-size: 0.85rem;
    }
    .zd-context-pill {
        display:inline-block; border:1px solid #C8A0A0; background:#EDD5D3; color:#842420;
        border-radius:999px; padding: 2px 10px; font-weight:700;
    }
    .zd-banner {
        background: #5A1410; border: 1.5px solid #842420; border-left: 5px solid #FF3030;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; color: #FDECEA; font-weight: 600;
    }
    .zd-section-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        color: #8A7060; border-bottom: 1.5px solid #D0BCAD; padding-bottom: 6px; margin-bottom: 14px;
    }
    .zd-stat-tile {
        background: #F5F0EB; border: 1.5px solid #D0BCAD; border-radius: 10px;
        padding: 12px 14px; box-shadow: 0 2px 8px rgba(44,24,16,0.05); min-height: 118px;
    }
    .zd-stat-title {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #8A7060;
    }
    .zd-stat-value {
        font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; color: #2C1810; line-height: 1.05;
        margin-top: 6px;
    }
    .zd-stat-subtitle { font-size: 0.74rem; color: #A89880; margin-top: 6px; }
    .zd-zone-card {
        background: #F5F0EB; border: 1.5px solid #D0BCAD; border-radius: 10px;
        padding: 14px 16px; margin-bottom: 10px; min-height: 180px;
    }
    .zd-zone-card.attacked {
        border-left: 4px solid #842420; box-shadow: 0 0 16px rgba(132,36,32,0.12);
    }
    .zd-zone-status { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: #842420; text-transform: uppercase; }
    .zd-zone-title { font-family: 'DM Mono', monospace; font-size: 1.65rem; font-weight: 700; color: #2C1810; margin-top: 8px; }
    .zd-zone-name { color: #5F524D; font-size: 0.95rem; margin-top: 2px; margin-bottom: 10px; }
    .zd-zone-meta { color: #2C1810; font-size: 0.92rem; margin-top: 4px; }
    .zd-pill-muted {
        border-radius: 999px; padding: 2px 10px; font-size: 0.68rem; font-weight: 700;
        background: #EDE5DC; color: #7A6A5A; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .zd-panel {
        background: #F5F0EB; border: 1.5px solid #D0BCAD; border-radius: 10px; padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(44,24,16,0.05);
    }
    /* Zone to Inspect selectbox — always light red */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #F5E8E6 !important;
        border-color: #C8A0A0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
        border-color: #842420 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] div,
    [data-testid="stSelectbox"] [data-baseweb="select"] p {
        color: #3D1A14 !important; font-weight: 600 !important;
        opacity: 1 !important; text-transform: none !important; letter-spacing: normal !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg { fill: #842420 !important; }
    /* Zone selectbox dropdown portal */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] ul[data-baseweb="menu"] {
        background-color: #F5E8E6 !important;
        border: 1.5px solid #C8A0A0 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="popover"] [role="option"] {
        background-color: #F5E8E6 !important;
        color: #3D1A14 !important;
        font-weight: 600 !important;
        border-bottom: 1.5px solid #D4968E !important;
    }
    [data-baseweb="popover"] [role="option"]:last-child { border-bottom: none !important; }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background-color: #EDD5D3 !important;
        color: #3D1A14 !important;
    }
    /* Download button — always light red */
    [data-testid="stDownloadButton"] > button {
        background-color: #F5E8E6 !important; border: 1.5px solid #C8A0A0 !important;
        color: #3D1A14 !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #EDD5D3 !important; border-color: #842420 !important;
    }
    /* Override dark multiselect in tab2 */
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div:focus-within {
        background-color: #FAF7F4 !important;
        border-color: #D0BCAD !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: #EDD5D3 !important;
        color: #842420 !important;
        border: 1px solid #C8A0A0 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg { color: #842420 !important; fill: #842420 !important; }
    [data-testid="stMultiSelect"] input { background-color: transparent !important; color: #2C1810 !important; }
    [data-testid="stMultiSelect"] label { color: #8A7060 !important; }
    [data-testid="stMultiSelect"] [data-baseweb="select"] svg { fill: #842420 !important; color: #842420 !important; }
    [data-testid="stMultiSelect"] [data-baseweb="select"] button svg { fill: #842420 !important; }
    [data-testid="stMultiSelect"] [role="button"] svg { fill: #842420 !important; }
    </style>
    """, unsafe_allow_html=True)

    snapshot = df_now.copy().sort_values("Zone_ID")

    def _zd_attacked_zone(_snap: pd.DataFrame) -> str:
        if _snap.empty:
            return ""
        if fire_active and fire_origin in _snap["Zone_ID"].astype(str).tolist():
            return fire_origin
        if "Event_Type" in _snap.columns:
            _fire_rows = _snap[_snap["Event_Type"] == "Fire Emergency"]
            if not _fire_rows.empty:
                _rank_cols = [c for c in ["Predicted_Hazard_Score", "Hazard_Level_Score", "Fire_Alarm_Triggered"] if c in _fire_rows.columns]
                if _rank_cols:
                    return str(_fire_rows.sort_values(_rank_cols, ascending=[False] * len(_rank_cols)).iloc[0]["Zone_ID"])
        _rank_cols = [c for c in ["Predicted_Hazard_Score", "Hazard_Level_Score", "DP_Total_Occupancy", "Total_Occupancy_Raw"] if c in _snap.columns]
        if _rank_cols:
            return str(_snap.sort_values(_rank_cols, ascending=[False] * len(_rank_cols)).iloc[0]["Zone_ID"])
        return str(_snap.iloc[0]["Zone_ID"])

    def _zd_fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    def stat_tile(title, value, subtitle='', accent='#842420'):
        st.markdown(
            f'''
            <div class="zd-stat-tile" style="border-top:4px solid {accent};">
                <div class="zd-stat-title">{title}</div>
                <div class="zd-stat-value">{value}</div>
                <div class="zd-stat-subtitle">{subtitle}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    attacked_zone_id = _zd_attacked_zone(snapshot)
    _bld_level = snapshot["Building_Level"].iloc[0] if "Building_Level" in snapshot.columns else fire_meta.get("level", "—")

    # ── Tick / time bar ───────────────────────────────────────────────────────
    _zd_live_color = "#842420" if not st.session_state.paused else "#8A6E15"
    _zd_live_label = "LIVE" if not st.session_state.paused else "PAUSED"
    _zd_live_bg    = "#EDD5D3" if not st.session_state.paused else "#F5EDD0"
    _zd_live_bd    = "#C8A0A0" if not st.session_state.paused else "#D4B840"
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;
                background:#F5F0EB;border:1.5px solid #D0BCAD;border-radius:8px;
                padding:7px 14px;margin-bottom:10px;'>
        <span style='font-family:"DM Mono",monospace;font-size:1.05rem;font-weight:600;color:#2C1810;'>
            {current_ts.strftime('%H:%M')}
            <span style='font-size:0.72rem;font-weight:400;color:#A89880;margin-left:6px;'>
                {current_ts.strftime('%a %d %b %Y')}
            </span>
        </span>
        <span style='font-size:0.68rem;color:#B0A090;'>
            Tick {st.session_state.tick + 1} / {n_ticks}
        </span>
        <span style='display:inline-flex;align-items:center;gap:5px;
                     background:{_zd_live_bg};border:1px solid {_zd_live_bd};
                     border-radius:999px;padding:3px 10px;
                     font-size:0.65rem;font-weight:700;color:{_zd_live_color};'>
            <span style='width:6px;height:6px;border-radius:50%;background:{_zd_live_color};
                         display:inline-block;'></span>
            {_zd_live_label} · 30s
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. Fire emergency banner (very top, most alerting) ───────────────────
    if fire_active:
        _alarm_total = int(snapshot["Fire_Alarm_Triggered"].fillna(0).astype(int).sum())
        st.markdown(f"""
        <div class='zd-banner'>
            FIRE EMERGENCY ACTIVE · Origin: {attacked_zone_id} — {ZONE_META.get(attacked_zone_id, attacked_zone_id)} ·
            {_alarm_total} alarm triggered · Wardens: initiate evacuation protocol
        </div>
        """, unsafe_allow_html=True)

    # ── 2. CTI context strip ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class='zd-context'>
        🏢 Centre for Technology &amp; Infusion (CTI) &nbsp;&nbsp;|&nbsp;&nbsp;
        FLOOR <b>{_bld_level}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
        INCIDENT ZONE <span class='zd-context-pill'>{attacked_zone_id} — {ZONE_META.get(attacked_zone_id, attacked_zone_id)}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Zone Status cards (all zones overview) ─────────────────────────────
    st.markdown("<div class='zd-section-label'>Zone Status</div>", unsafe_allow_html=True)
    _HZ_COLOR = {"Critical": "#842420", "High": "#B01808", "Moderate": "#C0A425", "Low": "#465030"}
    _zs = snapshot.sort_values("Zone_ID").copy()
    _zs["_label"]  = _zs["Zone_ID"].astype(str).apply(lambda z: f"{z}<br><sub>{ZONE_META.get(z, z)}</sub>")
    _zs["_hz_val"] = _zs.get("Predicted_Hazard_Score", _zs.get("Hazard_Level_Score", pd.Series([0]*len(_zs)))).fillna(0).clip(0, 100)
    _zs["_ev_val"] = _zs.get("Predicted_Evacuation_Priority_Score", _zs.get("Evacuation_Priority_Score", pd.Series([0]*len(_zs)))).fillna(0).clip(0, 100)
    _zs["_hz_lv"]  = _zs.apply(lambda r: str(r.get("Predicted_Hazard_Level", r.get("Hazard_Level_Rating", "Low"))), axis=1)
    _zs["_color"]  = _zs["_hz_lv"].map(_HZ_COLOR).fillna("#A89880")
    _zs["_status"] = _zs.apply(lambda r: "UNDER ATTACK" if str(r.get("Zone_ID","")) == attacked_zone_id else infer_zone_status(r), axis=1)
    _zs["_occ"]    = _zs.apply(lambda r: int(r.get("DP_Total_Occupancy", r.get("Total_Occupancy_Raw", 0))), axis=1)

    _col_chart, _col_table = st.columns([1.1, 0.9], gap="medium")

    with _col_chart:
        _fig_zs = go.Figure()
        _fig_zs.add_trace(go.Bar(
            x=_zs["_label"],
            y=_zs["_hz_val"],
            marker=dict(color=_zs["_color"].tolist(), opacity=0.88, line=dict(width=0)),
            customdata=list(zip(_zs["_status"], _zs["_occ"], _zs["_hz_lv"])),
            hovertemplate="<b>%{x}</b><br>Hazard: %{y:.1f}<br>Level: %{customdata[2]}<br>Status: %{customdata[0]}<br>Occupancy: ~%{customdata[1]}<extra></extra>",
            name="Hazard Score",
            showlegend=False,
        ))
        _fig_zs.add_trace(go.Scatter(
            x=_zs["_label"],
            y=_zs["_ev_val"],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="#2A5A80",
                        line=dict(width=1.5, color="#ffffff")),
            hovertemplate="<b>%{x}</b><br>Evac Priority: %{y:.1f}<extra></extra>",
            name="Evac Priority",
            showlegend=True,
        ))
        _fig_zs.update_layout(
            yaxis=dict(range=[0, 105], title="Score (0–100)", gridcolor="#EDE5DC",
                       tickfont=dict(size=10, family="DM Mono", color="#2C1810"),
                       title_font=dict(color="#2C1810"), zeroline=False),
            xaxis=dict(tickfont=dict(size=11, family="DM Sans", color="#2C1810"), automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10, color="#2C1810")),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            paper_bgcolor="#FDFAF7",
            plot_bgcolor="#FDFAF7",
            font=dict(family="DM Sans", color="#2C1810"),
            shapes=[
                dict(type="line", y0=75, y1=75, x0=-0.5, x1=len(_zs)-0.5,
                     line=dict(color="#842420", width=1.5, dash="dot")),
                dict(type="line", y0=50, y1=50, x0=-0.5, x1=len(_zs)-0.5,
                     line=dict(color="#C0A425", width=1.5, dash="dot")),
            ],
            annotations=[
                dict(y=76, x=len(_zs)-0.5, text="High", showarrow=False,
                     font=dict(size=9, color="#842420"), xanchor="right"),
                dict(y=51, x=len(_zs)-0.5, text="Moderate", showarrow=False,
                     font=dict(size=9, color="#C0A425"), xanchor="right"),
            ],
        )
        st.plotly_chart(_fig_zs, use_container_width=True, config={"displayModeBar": False})

    with _col_table:
        _ROUTE_PILL = {
            "Route Clear":   ("background:#E5ECDE;color:#465030;", "Clear"),
            "Partial Block": ("background:#F5EDD0;color:#8A6E15;", "Partial"),
            "Route Blocked": ("background:#EDD5D3;color:#842420;", "Blocked"),
        }
        _STATUS_BORDER = {
            "FIRE EMERGENCY": "#842420", "CRITICAL HAZARD": "#842420", "UNDER ATTACK": "#842420",
            "HIGH HAZARD": "#B01808", "ELEVATED": "#C0A425", "MONITORING": "#465030",
        }
        _strips = (
            "<style>"
            ".zs-wrap{border:1.5px solid #D0BCAD;border-radius:10px;overflow:hidden;background:#F5F0EB;}"
            ".zs-hdr{display:grid;grid-template-columns:38px 1fr 54px 54px 60px;"
            "padding:5px 12px;background:#EDE5DC;font-size:0.55rem;font-weight:700;"
            "letter-spacing:0.08em;text-transform:uppercase;color:#A89880;border-bottom:1.5px solid #D0BCAD;gap:6px;}"
            ".zs-row{display:grid;grid-template-columns:38px 1fr 54px 54px 60px;"
            "padding:7px 12px;border-bottom:1px solid #F0EAE4;border-left:4px solid #D0BCAD;"
            "align-items:center;gap:6px;}"
            ".zs-row:last-child{border-bottom:none;}"
            ".zs-row:hover{filter:brightness(0.96);}"
            "@keyframes zs-pulse{0%,100%{background:#EDD5D3;border-left-color:#C84040;box-shadow:none;}"
            "50%{background:#F5B8B4;border-left-color:#FF3030;box-shadow:inset 0 0 10px rgba(132,36,32,0.25);}}"
            ".zs-row-atk{animation:zs-pulse 1.8s ease-in-out infinite !important;}"
            ".zs-zid{font-family:monospace;font-size:0.72rem;font-weight:700;color:#465030;"
            "background:#E5ECDE;padding:1px 4px;border-radius:3px;text-align:center;}"
            ".zs-name{font-size:0.68rem;color:#2C1810;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
            ".zs-st{font-size:0.56rem;font-weight:700;letter-spacing:0.04em;margin-top:1px;}"
            ".zs-pill{font-size:0.54rem;font-weight:700;padding:1px 6px;border-radius:20px;white-space:nowrap;}"
            ".zs-num{font-family:monospace;font-size:0.72rem;text-align:right;color:#2C1810;}"
            "</style>"
            "<div class='zs-wrap'>"
            "<div class='zs-hdr'>"
            "<span>Zone</span><span>Status</span><span>Route</span>"
            "<span style='text-align:right'>Hazard</span>"
            "<span style='text-align:right'>Occ</span>"
            "</div>"
        )
        _HZ_PASTEL = {
            "Critical": "#EDD5D3", "High": "#F5DED8",
            "Moderate": "#F5EDD0", "Low": "#E5ECDE",
        }
        for _, _row in snapshot.sort_values("Zone_ID").iterrows():
            _zid  = str(_row.get("Zone_ID", ""))
            _atk  = _zid == attacked_zone_id and fire_active
            _st   = "UNDER ATTACK" if _atk else infer_zone_status(_row)
            _bc   = _STATUS_BORDER.get(_st, "#D0BCAD")
            _hz   = str(_row.get("Predicted_Hazard_Level", _row.get("Hazard_Level_Rating", "Low")))
            _occ  = int(_row.get("DP_Total_Occupancy", _row.get("Total_Occupancy_Raw", 0)))
            _rkey = route_label(str(_row.get("Exit_Route_Passable", "—")))
            _rst, _rl = _ROUTE_PILL.get(_rkey, ("background:#EDE5DC;color:#7A6A5A;", _rkey))
            _row_bg = _HZ_PASTEL.get(_hz, "#FAF7F4")
            _atk_cls = " zs-row-atk" if _atk and fire_active else ""
            _strips += (
                f"<div class='zs-row{_atk_cls}' style='border-left-color:{_bc};background:{_row_bg};'>"
                f"<span class='zs-zid'>{_zid}</span>"
                f"<div><div class='zs-name'>{ZONE_META.get(_zid, _zid)}</div>"
                f"<div class='zs-st' style='color:{_bc};'>{_st}</div></div>"
                f"<span><span class='zs-pill' style='{_rst}'>{_rl}</span></span>"
                f"<span class='zs-num' style='color:{_bc};font-weight:600;'>{_hz}</span>"
                f"<span class='zs-num'>~{_occ}</span>"
                f"</div>"
            )
        _strips += "</div>"
        st.markdown(_strips, unsafe_allow_html=True)

    # ── 4. Zone selector + hero banner ────────────────────────────────────────
    st.markdown("<div class='zd-section-label'>Zone Detail</div>", unsafe_allow_html=True)
    zone_options = snapshot["Zone_ID"].astype(str).tolist()
    selected_zone = st.selectbox(
        "Zone to inspect",
        zone_options,
        index=zone_options.index(attacked_zone_id) if attacked_zone_id in zone_options else 0,
        format_func=lambda z: f"{z} — {ZONE_META.get(z, z)}",
        key="zone_details_tab_zone_select",
    )

    zone_row     = snapshot[snapshot["Zone_ID"].astype(str) == str(selected_zone)].iloc[0]
    zone_history = df_scenario[df_scenario["Zone_ID"].astype(str) == str(selected_zone)].copy().sort_values("Timestamp")
    zone_status  = infer_zone_status(zone_row)
    error_margin = round((1.0 / max(float(epsilon), 0.01)) * np.log(20), 1)

    total_occ_raw    = max(float(zone_row.get("Total_Occupancy_Raw", 0) or 0), 1.0)
    vulnerable_ratio = (
        float(zone_row.get("Child_Count", 0))
        + float(zone_row.get("Elderly_Count", 0))
        + float(zone_row.get("Disabled_Count", 0))
    ) / total_occ_raw

    st.markdown(f"""
    <div class='zd-hero'>
        <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;'>
            <div>
                <div class='zd-hero-title'>Zone Command Center</div>
                <div class='zd-hero-sub'>{selected_zone} · {ZONE_META.get(selected_zone, selected_zone)} · {zone_row.get("Building_Level", "—")} · Snapshot {pd.Timestamp(current_ts).strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class='zd-status'>{zone_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    _left, _right = st.columns([1.08, 0.92])

    with _left:
        st.markdown("<div class='zd-panel'>", unsafe_allow_html=True)
        st.markdown("#### Trend Map")
        _trend_choices = [c for c in [
            "Predicted_Evacuation_Priority_Score",
            "Predicted_Hazard_Score",
            "DP_Total_Occupancy",
            "Total_Occupancy_Raw",
            "Temperature",
            "CO2",
            "PM2.5",
            "Smoke_Obscuration_Pct",
        ] if c in zone_history.columns]
        _default_trends = [c for c in ["Predicted_Evacuation_Priority_Score", "Predicted_Hazard_Score", "DP_Total_Occupancy"] if c in _trend_choices]
        _selected_trends = st.multiselect(
            "Trend metrics", _trend_choices,
            default=_default_trends[:3] if _default_trends else _trend_choices[:3],
            key="zone_details_trend_metrics",
        )
        if _selected_trends:
            _trend_df = zone_history[["Timestamp"] + _selected_trends].dropna()
            _trend_palette = [
                "#842420", "#B01808", "#C0A425", "#465030",
                "#5F524D", "#A89880", "#D0BCAD", "#8A7060",
            ]
            _trend_fig = go.Figure()
            for _i, _col in enumerate(_selected_trends):
                _trend_fig.add_trace(go.Scatter(
                    x=_trend_df["Timestamp"], y=_trend_df[_col],
                    mode="lines", name=_col,
                    line=dict(color=_trend_palette[_i % len(_trend_palette)], width=2),
                ))
            _trend_fig.update_layout(
                paper_bgcolor="#F5F0EB", plot_bgcolor="#FAF7F4",
                margin=dict(l=0, r=0, t=8, b=0),
                height=260,
                font=dict(family="DM Sans, sans-serif", color="#2C1810"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10, color="#2C1810"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    tickfont=dict(color="#5F524D", size=10),
                    gridcolor="#D0BCAD", linecolor="#D0BCAD",
                ),
                yaxis=dict(
                    tickfont=dict(color="#5F524D", size=10),
                    gridcolor="#D0BCAD", linecolor="#D0BCAD",
                ),
            )
            st.plotly_chart(_trend_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Select at least one metric.")
        st.markdown("</div>", unsafe_allow_html=True)

        _snap_occ   = _zd_fmt(zone_row.get("DP_Total_Occupancy", zone_row.get("Total_Occupancy_Raw", "—")))
        _snap_evac  = _zd_fmt(zone_row.get("Predicted_Evacuation_Priority_Score", "—"))
        _snap_hz    = _zd_fmt(zone_row.get("Predicted_Hazard_Score", "—"))
        _snap_err   = f"±{error_margin}"
        _snap_tiles = [
            ("Displayed Occupancy",  _snap_occ,  "Privacy-safe count",        "#465030"),
            ("Evacuation Priority",  _snap_evac, "Current urgency score",     "#B01808"),
            ("Hazard Score",         _snap_hz,   "ML hazard score",           "#842420"),
            ("Error Margin",         _snap_err,  "DP occupancy uncertainty",  "#465030"),
        ]
        _tiles_html = "".join(
            "<div style='background:#F5F0EB;border:1.5px solid #D0BCAD;border-top:4px solid " + _ac +
            ";border-radius:10px;padding:14px 14px 12px;display:flex;flex-direction:column;"
            "justify-content:space-between;min-height:110px;'>"
            "<div style='font-size:0.6rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;"
            "color:#8A7060;line-height:1.3;'>" + _tt + "</div>"
            "<div style='font-family:\"DM Mono\",monospace;font-size:1.6rem;font-weight:500;"
            "color:#2C1810;line-height:1.05;margin:6px 0 4px;'>" + _tv + "</div>"
            "<div style='font-size:0.68rem;color:#A89880;'>" + _ts + "</div>"
            "</div>"
            for _tt, _tv, _ts, _ac in _snap_tiles
        )
        _cmd_rows = [
            ("Zone Clearance", str(zone_row.get("Zone_Clearance_Status", "—"))),
            ("Route Status",   route_label(str(zone_row.get("Exit_Route_Passable", "—")))),
            ("Event Type",     str(zone_row.get("Event_Type", "—"))),
        ]
        _cmd_html = "".join(
            "<tr>"
            "<td style='font-size:0.68rem;font-weight:700;color:#8A7060;text-transform:uppercase;"
            "letter-spacing:0.07em;padding:9px 14px;border-bottom:1px solid #D0BCAD;"
            "white-space:nowrap;background:#EDE5DC;width:38%;'>" + _f + "</td>"
            "<td style='font-size:0.82rem;color:#2C1810;font-weight:500;padding:9px 14px;"
            "border-bottom:1px solid #D0BCAD;background:#F5F0EB;'>" + _v + "</td>"
            "</tr>"
            for _f, _v in _cmd_rows
        )
        _vul_pct   = int(vulnerable_ratio * 100)
        _vul_color = "#842420" if _vul_pct >= 60 else "#B01808" if _vul_pct >= 35 else "#C0A425" if _vul_pct >= 15 else "#465030"
        _snap_html = (
            "<div style='background:#F5F0EB;border:1.5px solid #D0BCAD;border-radius:10px;"
            "padding:16px 18px;margin-top:1rem;box-shadow:0 2px 8px rgba(44,24,16,0.05);'>"
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#8A7060;border-bottom:1.5px solid #D0BCAD;padding-bottom:6px;margin-bottom:14px;'>"
            "Occupancy &amp; Command Snapshot</div>"
            "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;'>"
            + _tiles_html +
            "</div>"
            "<div style='border:1.5px solid #D0BCAD;border-radius:8px;overflow:hidden;margin-bottom:12px;'>"
            "<table style='width:100%;border-collapse:collapse;'>"
            "<tbody>" + _cmd_html + "</tbody>"
            "</table></div>"
            "<div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
            "<span style='font-size:0.72rem;color:#8A7060;'>Vulnerable population ratio</span>"
            "<span style='font-family:\"DM Mono\",monospace;font-size:0.7rem;color:#5F524D;'>"
            + f"{vulnerable_ratio:.0%}" +
            "</span></div>"
            "<div style='background:#D0BCAD;border-radius:999px;height:6px;overflow:hidden;'>"
            "<div style='width:" + str(_vul_pct) + "%;height:100%;background:" + _vul_color +
            ";border-radius:999px;'></div></div>"
            "</div>"
        )
        st.markdown(_snap_html, unsafe_allow_html=True)

    with _right:
        st.markdown("<div class='zd-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='zd-section-label' style='margin-top:0;'>Event Log</div>", unsafe_allow_html=True)
        _event_cols = [c for c in ["Timestamp", "Event_Type", "Details"] if c in zone_history.columns]
        _event_log  = zone_history[_event_cols].copy().tail(20)
        if "Timestamp" in _event_log.columns:
            _event_log["Timestamp"] = pd.to_datetime(_event_log["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        st.download_button(
            label="↓  Download event log CSV",
            data=_event_log.to_csv(index=False).encode("utf-8"),
            file_name=f"event_log_{selected_base}_{selected_zone}.csv",
            mime="text/csv",
            key="zone_details_event_download",
            use_container_width=True,
        )
        # Styled HTML table — avoids Streamlit's dark-mode dataframe
        _ev_type_color = {
            "Fire Emergency":    ("#842420", "#EDD5D3"),
            "Routine Monitoring": ("#465030", "#E5ECDE"),
            "Evacuation":        ("#B01808", "#F5DED8"),
        }
        _ev_rows_html = ""
        for _, _er in _event_log.iterrows():
            _et  = str(_er.get("Event_Type", ""))
            _tc, _bg = _ev_type_color.get(_et, ("#5F524D", "#F5F0EB"))
            _det = str(_er.get("Details", ""))
            _ts  = str(_er.get("Timestamp", ""))
            _ev_rows_html += (
                f"<tr style='background:{_bg};'>"
                f"<td style='font-family:\"DM Mono\",monospace;font-size:0.7rem;color:#5F524D;"
                f"padding:6px 10px;border-bottom:1px solid #D0BCAD;white-space:nowrap;'>{_ts}</td>"
                f"<td style='font-size:0.7rem;font-weight:700;color:{_tc};"
                f"padding:6px 10px;border-bottom:1px solid #D0BCAD;white-space:nowrap;'>{_et}</td>"
                f"<td style='font-size:0.68rem;color:#2C1810;"
                f"padding:6px 10px;border-bottom:1px solid #D0BCAD;'>{_det}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div style='margin-top:8px;border:1.5px solid #D0BCAD;border-radius:8px;overflow:hidden;
                    max-height:420px;overflow-y:auto;'>
            <table style='width:100%;border-collapse:collapse;'>
                <thead>
                    <tr style='background:#EDE5DC;'>
                        <th style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
                                   color:#8A7060;padding:6px 10px;text-align:left;border-bottom:1.5px solid #D0BCAD;'>
                            Timestamp</th>
                        <th style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
                                   color:#8A7060;padding:6px 10px;text-align:left;border-bottom:1.5px solid #D0BCAD;'>
                            Event Type</th>
                        <th style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
                                   color:#8A7060;padding:6px 10px;text-align:left;border-bottom:1.5px solid #D0BCAD;'>
                            Details</th>
                    </tr>
                </thead>
                <tbody>{_ev_rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='zd-panel' style='margin-top:1rem;'>", unsafe_allow_html=True)
        st.markdown("#### Environment")
        _env_fields = [
            ("PM2.5",             zone_row.get("PM2.5",                  0),  0,   250),
            ("VOC",               zone_row.get("VOC",                    0),  0,  1000),
            ("Temperature",       zone_row.get("Temperature",            0), 15,    70),
            ("CO₂",               zone_row.get("CO2",                    0), 350, 2500),
            ("Humidity",          zone_row.get("Humidity",               0),  0,   100),
            ("Smoke Obscuration", zone_row.get("Smoke_Obscuration_Pct",  0),  0,   100),
        ]
        _env_html = ""
        for _env_label, _env_val, _env_low, _env_high in _env_fields:
            _pct = int(normalize(_env_val, _env_low, _env_high) * 100)
            _bar_color = "#842420" if _pct >= 75 else "#B01808" if _pct >= 45 else "#C0A425" if _pct >= 25 else "#465030"
            _env_html += f"""
            <div style='margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px;'>
                    <span style='font-size:0.78rem;font-weight:600;color:#2C1810;'>{_env_label}</span>
                    <span style='font-family:"DM Mono",monospace;font-size:0.72rem;color:#5F524D;'>{_zd_fmt(_env_val)}</span>
                </div>
                <div style='background:#D0BCAD;border-radius:999px;height:6px;overflow:hidden;'>
                    <div style='width:{_pct}%;height:100%;background:{_bar_color};border-radius:999px;'></div>
                </div>
            </div>"""
        st.markdown(_env_html, unsafe_allow_html=True)

        st.markdown("#### Narrative")
        st.markdown(
            f"<div style='font-size:1rem;line-height:1.65;color:#4A3A35;'>{zone_row.get('Details', 'No narrative available.')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    # ── Extra icons needed by this tab ────────────────────────────────────────
    _i3 = lambda path, w=14: (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}" '
        f'viewBox="0 0 16 16" fill="currentColor" '
        f'style="vertical-align:-2px;flex-shrink:0;">{path}</svg>'
    )
    _T3 = {
        **ICONS,
        "csv":      _i3('<path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2zM9.5 3A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5v2z"/>'),
        "share":    _i3('<path d="M13.5 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM11 2.5a2.5 2.5 0 1 1 .603 1.628l-6.718 3.12a2.499 2.499 0 0 1 0 1.504l6.718 3.12a2.5 2.5 0 1 1-.488.876l-6.718-3.12a2.5 2.5 0 1 1 0-3.256l6.718-3.12A2.5 2.5 0 0 1 11 2.5z"/>'),
        "print":    _i3('<path d="M2.5 8a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1z"/><path d="M5 1a2 2 0 0 0-2 2v2H2a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-1V3a2 2 0 0 0-2-2H5zM4 3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2H4V3zm1 5a2 2 0 0 0-2 2v1H2a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v-1a2 2 0 0 0-2-2H5zm7 2v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1z"/>'),
        "overview": _i3('<path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3zm8 0A1.5 1.5 0 0 1 10.5 1h3A1.5 1.5 0 0 1 15 2.5v3A1.5 1.5 0 0 1 13.5 7h-3A1.5 1.5 0 0 1 9 5.5v-3zm-8 8A1.5 1.5 0 0 1 2.5 9h3A1.5 1.5 0 0 1 7 10.5v3A1.5 1.5 0 0 1 5.5 15h-3A1.5 1.5 0 0 1 1 13.5v-3zm8 0A1.5 1.5 0 0 1 10.5 9h3a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-3A1.5 1.5 0 0 1 9 13.5v-3z"/>'),
        "report":   _i3('<path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2zM9.5 3A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5v2z"/><path d="M4.5 12.5A.5.5 0 0 1 5 12h6a.5.5 0 0 1 0 1H5a.5.5 0 0 1-.5-.5zm0-2A.5.5 0 0 1 5 10h6a.5.5 0 0 1 0 1H5a.5.5 0 0 1-.5-.5zm1.639-3.708 1.33.886 1.854-1.855a.25.25 0 0 1 .289-.047l1.888.974V8.5a.5.5 0 0 1-.5.5H5a.5.5 0 0 1-.5-.5V8s1.54-1.274 1.639-1.208z"/>'),
    }

    # ── CSS (new panel classes) ───────────────────────────────────────────────
    st.markdown("""
    <style>
    .meta-strip {
        background:#FFFFFF; border:1.5px solid #D0BCAD; border-radius:10px;
        padding:14px 20px; display:flex; align-items:center;
        justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:16px;
    }
    .meta-field { display:flex; flex-direction:column; }
    .meta-label { font-size:0.68rem; font-weight:700; letter-spacing:0.09em;
                  text-transform:uppercase; color:#A89880; margin-bottom:3px; }
    .meta-value { font-family:'DM Mono',monospace; font-size:0.88rem; font-weight:500; color:#2C1810; }
    .meta-divider { width:1px; height:36px; background:#D0BCAD; flex-shrink:0; }
    .stat-card {
        background:#FFFFFF; border:1.5px solid #D0BCAD; border-radius:10px;
        padding:14px 16px; display:flex; align-items:center; gap:14px; height:100%;
    }
    .stat-icon { width:38px; height:38px; border-radius:8px;
                 display:flex; align-items:center; justify-content:center; flex-shrink:0; }
    .stat-card-label { font-size:0.7rem; font-weight:700; letter-spacing:0.07em;
                       text-transform:uppercase; color:#A89880; }
    .stat-card-value { font-size:0.95rem; font-weight:600; color:#2C1810; margin-top:2px; }
    .panel-card { background:#FFFFFF; border:1.5px solid #D0BCAD; border-radius:10px;
                  padding:18px 20px; height:100%; }
    .panel-title { font-size:0.72rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                   color:#7A6A5A; border-bottom:1.5px solid #EDE5DC; padding-bottom:8px; margin-bottom:14px; }
    .bullet-list { list-style:none; padding:0; margin:0; }
    .bullet-list li { padding:5px 0; border-bottom:1px solid #F5F0E8; font-size:0.84rem;
                      color:#2C1810; display:flex; align-items:flex-start; gap:8px; line-height:1.4; }
    .bullet-list li:last-child { border-bottom:none; }
    .bullet-dot { width:6px; height:6px; border-radius:50%; background:#C08A80;
                  flex-shrink:0; margin-top:6px; }
    .tl-wrap { position:relative; }
    .tl-item { display:flex; align-items:flex-start; gap:10px; padding:7px 0; position:relative; }
    .tl-item:not(:last-child)::after {
        content:''; position:absolute; left:9px; top:24px;
        bottom:-7px; width:2px; background:#EDE5DC;
    }
    .tl-dot { width:20px; height:20px; border-radius:50%; border:2px solid #C08A80;
              background:#FFFFFF; flex-shrink:0; margin-top:1px; }
    .tl-time { font-family:'DM Mono',monospace; font-size:0.72rem;
               color:#A89880; min-width:38px; margin-top:3px; white-space:nowrap; }
    .tl-body { flex:1; }
    .tl-tag { display:inline-block; font-size:0.65rem; font-weight:700;
              letter-spacing:0.06em; text-transform:uppercase;
              padding:1px 7px; border-radius:20px; margin-bottom:2px; }
    .tl-desc { font-size:0.80rem; color:#2C1810; line-height:1.35; }
    .zone-tbl { width:100%; border-collapse:collapse; font-size:0.80rem; }
    .zone-tbl th { font-size:0.67rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase;
                   color:#A89880; padding:6px 10px; border-bottom:1.5px solid #D0BCAD; text-align:left; }
    .zone-tbl td { padding:7px 10px; border-bottom:1px solid #F0E8E2; color:#2C1810; vertical-align:middle; }
    .zone-tbl tr:last-child td { border-bottom:none; }
    .zone-tbl tr:hover td { background:#FAF5F0; }
    .notif-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .notif-item { display:flex; align-items:center; gap:8px; font-size:0.80rem;
                  color:#2C1810; padding:6px 8px; border-radius:7px; border:1px solid #EDE5DC; }
    .notif-check { width:16px; height:16px; border-radius:4px; border:1.5px solid #D0BCAD;
                   flex-shrink:0; display:flex; align-items:center; justify-content:center; }
    .action-btn { display:inline-flex; align-items:center; gap:7px; font-size:0.82rem;
                  font-weight:600; color:#465030; cursor:pointer; padding:7px 14px;
                  border-radius:7px; border:1.5px solid #AACBA0; background:#F0F5ED;
                  text-decoration:none; }
    .action-btn-primary { background:#465030 !important; color:#FFFFFF !important;
                          border-color:#465030 !important; }
    .action-divider { width:1px; height:28px; background:#D0BCAD; display:inline-block; }
    .status-badge { display:inline-flex; align-items:center; gap:5px; padding:3px 12px;
                    border-radius:20px; font-size:0.75rem; font-weight:700;
                    letter-spacing:0.05em; text-transform:uppercase; }

    @media print {
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stButton,
        .stSlider,
        .stSelectbox,
        footer,
        .no-print,
        .meta-actions,
        .action-divider {
            display: none !important;
        }

        @page {
            size: A4 portrait;
            margin: 1.2cm 1.4cm;
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="block-container"],
        .main .block-container,
        section.main {
            background: #FFFFFF !important;
            color: #2C1810 !important;
        }

        [data-testid="block-container"],
        [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
            width: 100% !important;
        }

        .panel-card,
        .meta-strip,
        .stat-card {
            break-inside: avoid;
            page-break-inside: avoid;
            box-shadow: none !important;
        }

        .zone-tbl {
            page-break-inside: auto;
        }

        .zone-tbl tr {
            page-break-inside: avoid;
            page-break-after: auto;
        }

        .zone-tbl thead {
            display: table-header-group;
        }

        .zone-tbl tfoot {
            display: table-footer-group;
        }

        a {
            text-decoration: none !important;
            color: inherit !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Style maps ─────────────────────────────────────────────────────────────
    _HAZARD_STYLE    = {"Low": ("#E5ECDE","#465030"), "Moderate": ("#F5EDD0","#8A6E15"),
                        "High": ("#F5DED8","#B01808"), "Critical": ("#EDD5D3","#842420")}
    _CLEAR_STYLE     = {"Cleared": ("#E5ECDE","#465030"), "Nearly Cleared": ("#F5EDD0","#8A6E15"),
                        "Occupied": ("#F5DED8","#B01808"), "Not Started": ("#EDE5DC","#7A6A5A")}
    _EVT_STYLE       = {
        "Fire Emergency":     ("#842420","#EDD5D3","#EDD5D3","#842420","FIRE"),
        "Smoke Spread Alert": ("#B01808","#F5DED8","#F5DED8","#B01808","SMOKE"),
        "Evacuation Update":  ("#8A6E15","#F5EDD0","#F5EDD0","#8A6E15","EVAC"),
        "All Clear":          ("#465030","#E5ECDE","#E5ECDE","#465030","CLEAR"),
        "Routine Monitoring": ("#7A6A5A","#EDE5DC","#EDE5DC","#7A6A5A","MON"),
        "System Check":       ("#2A5A80","#DDE8F0","#DDE8F0","#2A5A80","SYS"),
    }
    _EVT_DEFAULT     = ("#A89880","#EDE5DC","#EDE5DC","#7A6A5A","EVT")
    _HAZARD_ORDER    = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    _OUTCOME_STYLE   = {"Resolved": ("#E5ECDE","#465030"),
                        "Ongoing": ("#EDD5D3","#842420"),
                        "Partially Resolved": ("#F5EDD0","#8A6E15")}

    # ── Helper functions ───────────────────────────────────────────────────────
    def _col_val(row, *keys, default="—"):
        for k in keys:
            if k in row.index:
                v = row[k]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return v
        return default

    def _t3_pill(label, style_map, fallback=("#EDE5DC","#7A6A5A")):
        bg, color = style_map.get(str(label), fallback)
        return "<span class='pill' style='background:" + bg + ";color:" + color + ";'>" + str(label) + "</span>"

    def _build_csv(df_s, meta):
        want = ["Timestamp","Zone_ID","DP_Total_Occupancy","DP_Vulnerable_Count",
                "Hazard_Level_Rating","Zone_Clearance_Status",
                "Fire_Alarm_Triggered","Sprinkler_System_Status",
                "Exit_Route_Passable","Event_Type"]
        cols   = [c for c in want if c in df_s.columns]
        header = (
            "# INCIDENT SUMMARY REPORT\n"
            "# Incident ID: " + meta["incident_id"] + "\n"
            "# Building: " + meta["building"] + "\n"
            "# Generated: " + meta["generated_at"] + "\n"
            "# Epsilon (DP): " + str(meta["epsilon"]) + "\n#\n"
        )
        return header + df_s[cols].to_csv(index=False)

    # ── Data computation ───────────────────────────────────────────────────────
    from datetime import datetime as _dt

    # Incident overview must reflect the live simulation state, not the full future scenario.
    # End Time = current simulation timestamp.
    # Start Time = first fire alarm trigger up to the current simulation timestamp.
    _t_end = pd.Timestamp(current_ts)
    df_scenario["Timestamp"] = pd.to_datetime(df_scenario["Timestamp"])
    _df_report = df_scenario[df_scenario["Timestamp"] <= _t_end].copy()

    if "Fire_Alarm_Triggered" in _df_report.columns:
        _fire_alarm_records = _df_report[
            _df_report["Fire_Alarm_Triggered"].fillna(0).astype(float) > 0
        ].copy()
    else:
        _fire_alarm_records = pd.DataFrame()

    _alarm_count_total = int(len(_fire_alarm_records))
    _is_fire_incident = _alarm_count_total > 0

    if _is_fire_incident:
        _t_start = pd.Timestamp(_fire_alarm_records["Timestamp"].min())
    else:
        _t_start = _t_end

    # Incident progress window = first alarm trigger through current simulation time.
    # Highest Hazard Zone should be calculated from this window only.
    if _is_fire_incident:
        _df_incident_progress = _df_report[
            (_df_report["Timestamp"] >= _t_start) &
            (_df_report["Timestamp"] <= _t_end)
        ].copy()
    else:
        _df_incident_progress = _df_report.copy()

    np.random.seed(42)
    _df_final = df_scenario[df_scenario["Timestamp"] == _t_end].copy()
    if "Total_Occupancy_Raw" in _df_final.columns:
        _raw_t   = _df_final["Total_Occupancy_Raw"].clip(lower=1)
        _df_final["DP_Total_Occupancy"] = laplace_noise(_df_final["Total_Occupancy_Raw"], epsilon)
        _dp_t    = _df_final["DP_Total_Occupancy"]
        for _sub in ["Child_Count","Elderly_Count","Disabled_Count"]:
            _dpk = "DP_" + _sub
            _df_final[_dpk] = (
                (_df_final[_sub] / _raw_t * _dp_t).clip(0).round().astype(int)
                if _sub in _df_final.columns else 0
            )
        _df_final["DP_Vulnerable_Count"] = (
            _df_final["DP_Child_Count"] + _df_final["DP_Elderly_Count"] + _df_final["DP_Disabled_Count"]
        )

    _total_occ_final  = int(_df_final["DP_Total_Occupancy"].sum()) if "DP_Total_Occupancy" in _df_final.columns else 0
    _total_vuln_final = int(_df_final["DP_Vulnerable_Count"].sum()) if "DP_Vulnerable_Count" in _df_final.columns else 0
    _cleared_count    = int((_df_final["Zone_Clearance_Status"] == "Cleared").sum()) if "Zone_Clearance_Status" in _df_final.columns else 0
    _total_zones      = len(_df_final)
    _peak_occ         = int(_df_report.groupby("Timestamp")["Total_Occupancy_Raw"].sum().max()) if "Total_Occupancy_Raw" in _df_report.columns and not _df_report.empty else 0
    _sprinkler_on     = bool("Sprinkler_System_Status" in _df_report.columns and _df_report["Sprinkler_System_Status"].eq("Activated").any())

    if (
        not _df_incident_progress.empty
        and "Predicted_Hazard_Score" in _df_incident_progress.columns
        and "Zone_ID" in _df_incident_progress.columns
    ):
        # Use peak hazard during the incident so far, not average over the full scenario.
        _highest_hz_zone = str(
            _df_incident_progress.groupby("Zone_ID")["Predicted_Hazard_Score"].max().idxmax()
        )
    elif (
        not _df_incident_progress.empty
        and "Hazard_Level_Rating" in _df_incident_progress.columns
        and "Zone_ID" in _df_incident_progress.columns
    ):
        _highest_hz_zone = str(
            _df_incident_progress.groupby("Zone_ID")["Hazard_Level_Rating"]
            .apply(lambda s: max([_HAZARD_ORDER.get(str(x), 0) for x in s.dropna()], default=0))
            .idxmax()
        )
    else:
        _highest_hz_zone = list(ZONE_META.keys())[0]
    _highest_hz_name = ZONE_META.get(_highest_hz_zone, _highest_hz_zone)

    _dur_min      = int((_t_end - _t_start).total_seconds()) // 60
    _dur_str      = (f"{_dur_min // 60}h {_dur_min % 60:02d}m" if _dur_min >= 60 else f"{_dur_min}m")
    _inc_type     = "Fire Emergency" if _is_fire_incident else "Routine Monitoring"
    _generated_at = _dt.now().strftime("%d %b %Y, %H:%M")
    _final_outcome= (
        "Resolved" if _cleared_count == _total_zones and _total_zones > 0
        else "Ongoing" if _cleared_count == 0
        else "Partially Resolved"
    )
    _out_bg, _out_color = _OUTCOME_STYLE[_final_outcome]
    _report_meta  = {"incident_id": selected_base, "building": "CTI — La Trobe University",
                     "generated_at": _generated_at, "epsilon": epsilon}
    _csv_b64      = base64.b64encode(_build_csv(df_scenario, _report_meta).encode()).decode()

    # ── Header ─────────────────────────────────────────────────────────────────
    _col_h, _col_bc = st.columns([3, 2])
    with _col_h:
        st.markdown("<h1>Incident Report</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#A89880;font-size:0.85rem;margin-top:-10px;'>"
            "Post-incident report — operational summary, final zone outcomes, exportable records</p>",
            unsafe_allow_html=True,
        )
    with _col_bc:
        st.markdown(
            "<div style='padding-top:20px;text-align:right;font-size:0.8rem;color:#A89880;'>"
            "Overview &nbsp;/&nbsp; Reports &nbsp;/&nbsp;"
            "<span style='color:#2C1810;font-weight:600;'>Incident Summary</span></div>",
            unsafe_allow_html=True,
        )

    # ── Meta strip ─────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='meta-strip'>"
        "<div class='meta-field'>"
        "<span class='meta-label'>" + _T3["building"] + " Building</span>"
        "<span class='meta-value'>CTI — La Trobe</span></div>"
        "<div class='meta-divider'></div>"
        "<div class='meta-field'>"
        "<span class='meta-label'>" + _T3["report"] + " Incident ID</span>"
        "<span class='meta-value'>" + selected_base + "</span></div>"
        "<div class='meta-divider'></div>"
        "<div class='meta-field'>"
        "<span class='meta-label'>" + _T3["clock"] + " Generated At</span>"
        "<span class='meta-value'>" + _generated_at + "</span></div>"
        "<div class='meta-divider'></div>"
        "<div class='meta-field'>"
        "<span class='meta-label'>" + _T3["check"] + " Report Status</span>"
        "<span class='meta-value'>"
        "<span class='status-badge' style='background:" + _out_bg + ";color:" + _out_color + ";'>"
        + _final_outcome + "</span></span></div>"
        "<div class='meta-actions no-print'>"
        "<a class='action-btn' href='data:text/csv;base64," + _csv_b64 + "' download='" + selected_base + "_report.csv'>"
        + _T3["csv"] + " CSV</a></div></div>",
        unsafe_allow_html=True,
    )

    # ── Stat cards (6 columns) ─────────────────────────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:12px;'>Incident Overview</div>", unsafe_allow_html=True)
    _stat_data = [
        (_T3["fire"],     "#EDD5D3", "#842420", "Incident Type",       _inc_type),
        (_T3["clock"],    "#F5EDD0", "#8A6E15", "Start Time",          _t_start.strftime("%H:%M · %d %b")),
        (_T3["clock"],    "#EDE5DC", "#7A6A5A", "End Time",            _t_end.strftime("%H:%M · %d %b")),
        (_T3["warning"],  "#DDE8F0", "#2A5A80", "Duration",            _dur_str),
        (_T3["building"], "#F5DED8", "#B01808", "Highest Hazard Zone", _highest_hz_zone + " — " + _highest_hz_name),
        (_T3["check"],    _out_bg,   _out_color,"Final Outcome",       _final_outcome),
    ]
    _scols = st.columns(6)
    for _sc, (_icon, _bg, _color, _label, _value) in zip(_scols, _stat_data):
        with _sc:
            st.markdown(
                "<div class='stat-card'>"
                "<div class='stat-icon' style='background:" + _bg + ";color:" + _color + ";'>" + _icon + "</div>"
                "<div>"
                "<div class='stat-card-label'>" + _label + "</div>"
                "<div class='stat-card-value' style='color:" + _color + ";'>" + _value + "</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

    # ── Four panels ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _p1, _p2, _p3, _p4 = st.columns([1.1, 1.1, 1.6, 1.2], gap="medium")

    # Panel 1 · Incident Summary
    with _p1:
        if is_fire:
            _bullets = [
                "Fire emergency at " + _t_start.strftime("%H:%M") + " on " + _t_start.strftime("%d %b %Y") + ".",
                "Origin: <b>" + fire_origin + " — " + ZONE_META.get(fire_origin,"") + "</b> (" + fire_meta.get("level","") + ").",
                "Peak building occupancy: <b>" + str(_peak_occ) + " persons</b> (DP-protected).",
                "<b>" + str(_alarm_count_total) + "</b> fire alarm trigger(s) recorded.",
                "<b>" + str(_cleared_count) + " of " + str(_total_zones) + "</b> zones confirmed cleared.",
                "Vulnerable persons at final snapshot: <b>" + str(_total_vuln_final) + "</b>.",
            ]
        else:
            _bullets = [
                "Routine monitoring from " + _t_start.strftime("%H:%M") + " to " + _t_end.strftime("%H:%M") + ".",
                "Building: Centre for Technology &amp; Infusion, La Trobe.",
                "Peak occupancy: <b>" + str(_peak_occ) + " persons</b> (DP-protected).",
                "No fire alarms triggered during this session.",
                "All <b>" + str(_total_zones) + "</b> zones at Low or Moderate hazard.",
                "DP budget ε = " + str(epsilon) + " applied throughout.",
            ]
        _li = "".join("<li><span class='bullet-dot'></span><span>" + b + "</span></li>" for b in _bullets)
        st.markdown("<div class='panel-card'><div class='panel-title'>Incident Summary</div><ul class='bullet-list'>" + _li + "</ul></div>", unsafe_allow_html=True)

    # Panel 2 · Final Evacuation Outcome
    with _p2:
        _evac_items = []
        for _, _row in _df_final.sort_values("Zone_ID").iterrows():
            _zid2  = str(_row["Zone_ID"])
            _zname2= ZONE_META.get(_zid2, _zid2)
            _st2   = str(_col_val(_row, "Zone_Clearance_Status", default="Unknown"))
            _occ2  = int(_col_val(_row, "DP_Total_Occupancy", default=0))
            _, _dc = _CLEAR_STYLE.get(_st2, ("#EDE5DC","#7A6A5A"))
            _evac_items.append(
                "<li><span class='bullet-dot' style='background:" + _dc + ";'></span>"
                "<span><b>" + _zid2 + "</b> " + _zname2 + " — "
                "<span style='color:" + _dc + ";font-weight:600;'>" + _st2 + "</span>"
                " · " + str(_occ2) + " remaining</span></li>"
            )
        st.markdown(
            "<div class='panel-card'><div class='panel-title'>Final Evacuation Outcome</div>"
            "<ul class='bullet-list'>" + "".join(_evac_items) + "</ul>"
            "<div style='margin-top:12px;padding-top:10px;border-top:1px solid #EDE5DC;"
            "font-size:0.76rem;color:#A89880;'>"
            + _T3["lock"] + " Counts are DP-protected (ε = " + str(epsilon) + ")"
            "</div></div>",
            unsafe_allow_html=True,
        )

    
    # Panel 3 · Final Zone Status Table
    with _p3:
        _tbl_rows = []
        for _, _row in _df_final.sort_values("Zone_ID").iterrows():
            _zid3  = str(_row["Zone_ID"])
            _st3   = str(_col_val(_row, "Zone_Clearance_Status", default="Unknown"))
            _hz3   = str(_col_val(_row, "Predicted_Hazard_Level", "Hazard_Level_Rating", default="Low"))
            _occ3  = int(_col_val(_row, "DP_Total_Occupancy", default=0))
            _sw_raw= _col_val(_row, "Last_Manual_Sweep_Time", default=None)
            if _sw_raw is None or str(_sw_raw).strip() in ("—","","nan","None"):
                _sweep = "—"
            else:
                try:    _sweep = pd.to_datetime(str(_sw_raw)).strftime("%H:%M")
                except: _sweep = str(_sw_raw)
            _sb, _sc = _CLEAR_STYLE.get(_st3, ("#EDE5DC","#7A6A5A"))
            _hb, _hc = _HAZARD_STYLE.get(_hz3, ("#EDE5DC","#7A6A5A"))
            _tbl_rows.append(
                "<tr>"
                "<td style='white-space:nowrap;'>"
                "<span style='font-family:monospace;font-size:0.78rem;font-weight:700;"
                "color:#465030;background:#E5ECDE;padding:1px 6px;border-radius:4px;'>"
                + _zid3 +
                "</span><span style='font-size:0.75rem;color:#7A6A5A;margin-left:5px;'>"
                + ZONE_META.get(_zid3,_zid3) + "</span></td>"
                "<td><span class='pill' style='background:" + _sb + ";color:" + _sc + ";'>" + _st3 + "</span></td>"
                "<td><span class='pill' style='background:" + _hb + ";color:" + _hc + ";'>" + _hz3 + "</span></td>"
                "<td style='font-family:monospace;font-size:0.8rem;text-align:right;'>" + str(_occ3) + "</td>"
                "<td style='font-family:monospace;font-size:0.76rem;color:#A89880;text-align:center;'>" + _sweep + "</td>"
                "</tr>"
            )
        st.markdown(
            "<div class='panel-card' style='padding:16px 14px;'>"
            "<div class='panel-title'>Final Zone Status Table</div>"
            "<div style='overflow-x:auto;overflow-y:auto;max-height:280px;"
            "border-radius:6px;border:1px solid #EDE5DC;'>"
            "<table class='zone-tbl' style='min-width:480px;width:100%;border-collapse:collapse;'>"
            "<thead><tr style='position:sticky;top:0;background:#FDFAF7;z-index:1;'>"
            "<th>Zone</th><th>Status</th><th>Hazard</th>"
            "<th style='text-align:right;'>Occupants (DP)</th>"
            "<th style='text-align:center;'>Last Sweep</th>"
            "</tr></thead><tbody>" + "".join(_tbl_rows) + "</tbody></table></div></div>",
            unsafe_allow_html=True,
        )

    # Panel 4 · Incident Timeline
    with _p4:
        _tl_items = []
        if "Event_Type" in df_scenario.columns:
            _step = max(1, len(timestamps) // 8)
            _seen = set()
            _evts = []
            for _ts in timestamps[::_step]:
                for _et in df_scenario[df_scenario["Timestamp"] == _ts]["Event_Type"].dropna().unique():
                    _et = str(_et).strip()
                    if _et:
                        _k = (pd.Timestamp(_ts).strftime("%H%M"), _et)
                        if _k not in _seen:
                            _seen.add(_k)
                            _evts.append((pd.Timestamp(_ts), _et))
            for _ts, _et in sorted(_evts, key=lambda x: x[0])[-6:]:
                _db, _dbg, _tb, _tc, _tl = _EVT_STYLE.get(_et, _EVT_DEFAULT)
                _tl_items.append(
                    "<div class='tl-item'>"
                    "<div class='tl-dot' style='border-color:" + _db + ";background:" + _dbg + ";'></div>"
                    "<div class='tl-time'>" + _ts.strftime("%H:%M") + "</div>"
                    "<div class='tl-body'>"
                    "<span class='tl-tag' style='background:" + _tb + ";color:" + _tc + ";'>" + _tl + "</span>"
                    "<div class='tl-desc'>" + _et + "</div>"
                    "</div></div>"
                )
        if not _tl_items:
            _tl_items.append("<p style='font-size:0.8rem;color:#A89880;margin:0;'>No events available.</p>")
        st.markdown(
            "<div class='panel-card'><div class='panel-title'>Incident Timeline</div>"
            "<div class='tl-wrap'>" + "".join(_tl_items) + "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Bottom row ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _bot_l, _bot_r = st.columns(2, gap="medium")

    # Notifications
    with _bot_l:
        if is_fire:
            _notifs = [
                (True,                         "Fire alarm triggered"),
                (True,                         "Evacuation protocol initiated"),
                (_alarm_count_total > 0,        str(_alarm_count_total) + " zone alarm(s) activated"),
                (True,                         "Warden zones swept"),
                (_sprinkler_on,                "Sprinkler system activated"),
                (_cleared_count == _total_zones,"All zones confirmed cleared"),
            ]
        else:
            _notifs = [
                (True,  "Routine monitoring session active"),
                (True,  "Sensor data recorded across all zones"),
                (True,  "DP noise applied to all occupancy counts"),
                (False, "No fire alarms triggered"),
                (False, "Evacuation not required"),
                (True,  "Session log archived"),
            ]
        _TICK_SVG = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' "
            "viewBox='0 0 16 16' fill='#465030'>"
            "<path d='M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0"
            " l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293"
            " l6.646-6.647a.5.5 0 0 1 .708 0z'/></svg>"
        )
        _grid = []
        for _chk, _lbl in _notifs:
            _chk    = bool(_chk)
            _ibg    = "#E5ECDE" if _chk else "#FDFAF7"
            _ibdr   = "#AACBA0" if _chk else "#EDE5DC"
            _cbg    = "#E5ECDE" if _chk else "#FDFAF7"
            _cbdr   = "#465030" if _chk else "#D0BCAD"
            _tick   = _TICK_SVG if _chk else ""
            _grid.append(
                "<div class='notif-item' style='background:" + _ibg + ";border-color:" + _ibdr + ";'>"
                "<div class='notif-check' style='background:" + _cbg + ";border-color:" + _cbdr + ";'>" + _tick + "</div>"
                "<span>" + _lbl + "</span></div>"
            )
        st.markdown(
            "<div class='panel-card'><div class='panel-title'>Notifications / Actions Observed</div>"
            "<div class='notif-grid'>" + "".join(_grid) + "</div></div>",
            unsafe_allow_html=True,
        )

    # Report Actions

    with _bot_r:
        # ── Build downloadable PDF via fpdf2 ──────────────────────────────────
        def _build_pdf_bytes(df_s, meta, zone_meta, is_fire_inc,
                             inc_type, t_start, t_end, dur_str,
                             peak_occ, alarm_count, cleared, total_z,
                             vuln_final, highest_hz_zone, highest_hz_name,
                             final_outcome, sprinkler):
            try:
                from fpdf import FPDF
            except ImportError:
                return None

            # ── Unicode → latin-1 sanitizer (fpdf v1 only supports latin-1) ──
            _UNICODE_MAP = {
                "\u2014": "-",   # em dash         —
                "\u2013": "-",   # en dash         –
                "\u00b7": ".",   # middle dot      ·
                "\u03b5": "e",   # epsilon         ε
                "\u2082": "2",   # subscript 2     ₂
                "\u2019": "'",   # right single quote
                "\u2018": "'",   # left single quote
                "\u201c": '"',   # left double quote
                "\u201d": '"',   # right double quote
                "\u00e9": "e",   # é
                "\u00e8": "e",   # è
                "\u00e0": "a",   # à
                "\u00fc": "u",   # ü
                "\u00f6": "o",   # ö
                "\u00e4": "a",   # ä
            }
            def _safe(text):
                s = str(text)
                for uni, asc in _UNICODE_MAP.items():
                    s = s.replace(uni, asc)
                # Drop any remaining non-latin-1 characters
                return s.encode("latin-1", errors="ignore").decode("latin-1")

            pdf = FPDF(orientation="P", unit="mm", format="A4")
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title block
            pdf.set_fill_color(70, 80, 48)
            pdf.rect(0, 0, 210, 22, "F")
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(12, 5)
            pdf.cell(0, 12, "Incident Report - Live Warden Dashboard", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_xy(12, 15)
            pdf.cell(0, 6, "Privacy-First AI for Emergency Response - Centre for Technology and Infusion, La Trobe University")
            pdf.ln(10)

            # Meta strip
            pdf.set_text_color(44, 24, 16)
            fields = [
                ("Incident ID", meta["incident_id"]),
                ("Building",    meta["building"]),
                ("Generated",   meta["generated_at"]),
                ("DP Budget e", str(meta["epsilon"])),
                ("Outcome",     final_outcome),
            ]
            col_w = 38
            for label, _ in fields:
                pdf.set_font("Helvetica", "B", 7)
                pdf.set_text_color(168, 152, 128)
                pdf.cell(col_w, 5, _safe(label.upper()), ln=False)
            pdf.ln(5)
            for _, value in fields:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(44, 24, 16)
                pdf.cell(col_w, 6, _safe(str(value))[:18], ln=False)
            pdf.ln(10)

            def section(title):
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(168, 152, 128)
                pdf.set_draw_color(208, 188, 173)
                pdf.cell(0, 5, _safe(title.upper()), ln=True, border="B")
                pdf.ln(2)

            section("Incident Overview")
            overview_rows = [
                ("Incident Type",      inc_type),
                ("Start Time",         t_start.strftime("%H:%M - %d %b %Y")),
                ("End Time",           t_end.strftime("%H:%M - %d %b %Y")),
                ("Duration",           dur_str),
                ("Peak Occupancy",     str(peak_occ) + " persons (DP-protected)"),
                ("Highest Hazard Zone",highest_hz_zone + " - " + highest_hz_name),
                ("Alarm Records",     str(alarm_count) + " active alarm readings" if is_fire_inc else "None"),
                ("Sprinkler System",   "Activated" if sprinkler else "Not activated"),
                ("Zones Cleared",      str(cleared) + " of " + str(total_z)),
                ("Vulnerable Persons", str(vuln_final) + " (final snapshot, DP)"),
            ]
            for k, v in overview_rows:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(44, 24, 16)
                pdf.cell(55, 6, _safe(k + ":"), ln=False)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(0, 6, _safe(str(v)), ln=True)
            pdf.ln(4)

            # ── Environmental Factors by Zone - Final Snapshot ─────────────────────
            section("Environmental Factors by Zone - Final Snapshot")

            env_cols = [
                ("Temp", "Temperature", "C"),
                ("CO2", "CO2", "ppm"),
                ("PM2.5", "PM2.5", ""),
                ("VOC", "VOC", ""),
                ("Humidity", "Humidity", "%"),
                ("Smoke", "Smoke_Obscuration_Pct", "%"),
            ]

            t_end_ts = df_s["Timestamp"].max()
            df_env_final = df_s[df_s["Timestamp"] == t_end_ts].copy()

            available_env_cols = [
                (label, col, unit)
                for label, col, unit in env_cols
                if col in df_env_final.columns
            ]

            if available_env_cols:
                env_headers = ["Zone", "Name"] + [label for label, _, _ in available_env_cols]

                # Total usable PDF width is around 190mm
                base_widths = [16, 34]
                env_width = int((190 - sum(base_widths)) / len(available_env_cols))
                env_col_widths = base_widths + [env_width] * len(available_env_cols)

                pdf.set_font("Helvetica", "B", 7)
                pdf.set_fill_color(237, 229, 220)
                pdf.set_text_color(90, 70, 60)

                for h, w in zip(env_headers, env_col_widths):
                    pdf.cell(w, 6, _safe(h)[:14], border=1, fill=True, ln=False)
                pdf.ln()

                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(44, 24, 16)

                for _, row in df_env_final.sort_values("Zone_ID").iterrows():
                    zid = str(row.get("Zone_ID", "-"))
                    zname = zone_meta.get(zid, zid)

                    row_values = [zid, _safe(zname)[:16]]

                    for label, col, unit in available_env_cols:
                        value = row.get(col, "-")

                        if pd.isna(value):
                            formatted = "-"
                        else:
                            try:
                                value_float = float(value)

                                if col == "Temperature":
                                    formatted = f"{value_float:.1f} {unit}"
                                elif col == "CO2":
                                    formatted = f"{value_float:.0f} {unit}"
                                elif col == "PM2.5":
                                    formatted = f"{value_float:.1f}"
                                elif col == "VOC":
                                    formatted = f"{value_float:.0f}"
                                elif col == "Humidity":
                                    formatted = f"{value_float:.0f}{unit}"
                                elif col == "Smoke_Obscuration_Pct":
                                    formatted = f"{value_float:.1f}{unit}"
                                else:
                                    formatted = str(value)
                            except Exception:
                                formatted = str(value)

                        row_values.append(_safe(formatted))

                    for val, w in zip(row_values, env_col_widths):
                        pdf.cell(w, 6, str(val)[:14], border=1, ln=False)
                    pdf.ln()

                pdf.ln(4)

                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(122, 106, 90)
                pdf.multi_cell(
                    0,
                    5,
                    _safe(
                        "Environmental readings are taken from the same zone-level sensor data used in the dashboard overview. "
                        "Values shown here represent the final incident snapshot for each zone."
                    ),
                )
                pdf.ln(3)

            else:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(122, 106, 90)
                pdf.cell(0, 6, "No environmental sensor data available for this scenario.", ln=True)
                pdf.ln(4)

            section("Final Zone Status Table")
            headers = ["Zone", "Name", "Status", "Hazard", "Occupants (DP)", "Last Sweep"]
            col_ws  = [18, 36, 34, 26, 32, 28]
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_fill_color(237, 229, 220)
            pdf.set_text_color(90, 70, 60)
            for h, w in zip(headers, col_ws):
                pdf.cell(w, 6, h, border=1, fill=True, ln=False)
            pdf.ln()

            t_end_ts = df_s["Timestamp"].max()
            df_final_pdf = df_s[df_s["Timestamp"] == t_end_ts].copy()
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(44, 24, 16)
            for _, row in df_final_pdf.sort_values("Zone_ID").iterrows():
                zid   = str(row.get("Zone_ID", ""))
                zname = zone_meta.get(zid, zid)
                st_   = str(row.get("Zone_Clearance_Status", "-"))
                hz    = str(row.get("Hazard_Level_Rating", row.get("Predicted_Hazard_Level", "-")))
                occ   = str(int(row.get("DP_Total_Occupancy", 0)))
                sw    = row.get("Last_Manual_Sweep_Time", None)
                if sw is None or str(sw).strip() in ("\u2014", "-", "", "nan", "None"):
                    sweep = "-"
                else:
                    try:    sweep = pd.to_datetime(str(sw)).strftime("%H:%M")
                    except: sweep = _safe(str(sw))
                for val, w in zip([zid, _safe(zname)[:18], _safe(st_)[:18], _safe(hz), occ, sweep], col_ws):
                    pdf.cell(w, 6, str(val), border=1, ln=False)
                pdf.ln()
            pdf.ln(4)

            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(122, 106, 90)
            pdf.multi_cell(
                0, 5,
                _safe(
                    f"All occupancy figures are differentially private (Laplace mechanism, epsilon={meta['epsilon']}). "
                    "No personal identifiers are stored or displayed. "
                    "This report is generated for authorised warden use only."
                ),
            )
            raw = pdf.output()
            
            
            raw = pdf.output()
            if isinstance(raw, bytearray):
                return bytes(raw)
            if isinstance(raw, bytes):
                return raw    
            return raw.encode("latin-1")

        import subprocess, sys
        try:
            import fpdf  # noqa
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "fpdf2", "--quiet", "--break-system-packages"],
                capture_output=True,
            )

        _pdf_bytes = _build_pdf_bytes(
            df_scenario, _report_meta, ZONE_META, is_fire,
            _inc_type, _t_start, _t_end, _dur_str,
            _peak_occ, _alarm_count_total, _cleared_count, _total_zones,
            _total_vuln_final, _highest_hz_zone, _highest_hz_name,
            _final_outcome, _sprinkler_on,
        )
        _pdf_b64 = base64.b64encode(_pdf_bytes).decode() if _pdf_bytes else None

        _pdf_btn = (
            "<a class='action-btn action-btn-primary' "
            "href='data:application/pdf;base64," + _pdf_b64 + "' "
            "download='" + selected_base + "_incident_report.pdf'>"
            + _T3["print"] + " Download PDF</a>"
            if _pdf_b64
            else
            "<span style='font-size:0.78rem;color:#A89880;'>PDF unavailable (fpdf2 not installed)</span>"
        )
    with _bot_r:
    # Report Actions code     

        st.markdown(
            "<div class='panel-card'><div class='panel-title'>Report Actions</div>"
            "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-top:4px;'>"
            + _pdf_btn +
            "<span class='action-divider'></span>"
            "<a class='action-btn' href='data:text/csv;base64," + _csv_b64 + "' download='" + selected_base + "_report.csv'>"
            + _T3["csv"] + " Export CSV</a>"
            "<span class='action-divider'></span>"
            "<span style='font-size:0.78rem;color:#A89880;display:flex;align-items:center;gap:5px;'>"
            + _T3["lock"] + " DP ε = " + str(epsilon) +
            "</span></div>"
            "<div style='margin-top:14px;padding:10px 12px;background:#F5F0EC;"
            "border-radius:7px;border:1px solid #E8DDD4;font-size:0.78rem;color:#7A6A5A;'>"
            "<b>Tip:</b> Click <b>Download PDF</b> for a clean incident report, "
            "or <b>Export CSV</b> for raw DP-protected data."
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Screen footer ──────────────────────────────────────────────────────────
    st.markdown(
        "<div class='no-print' style='margin-top:30px;padding-top:14px;border-top:1.5px solid #D0BCAD;"
        "font-size:0.72rem;color:#CCCCCC;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;'>"
        "<span>Privacy-First AI for Emergency Response · Centre for Technology and Infusion, La Trobe University</span>"
        "<span>All occupancy data is differentially private. No personal identifiers are stored or displayed.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

