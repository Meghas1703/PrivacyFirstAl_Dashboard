import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
np.random.seed(42)
Faker.seed(42)


@dataclass(frozen=True)
class Zone:
    zone_id: str
    zone_name: str
    base_occ: int


@dataclass(frozen=True)
class Scenario:
    incident_id: str
    building_level: str
    start_time: datetime
    end_time: datetime
    fire_start_time: datetime | None
    fire_origin_zone: str | None
    adjacent_zones: set[str]
    event_family: str


TIME_STEP_MINUTES = 5
BUILDING_NAME = "Synthetic Privacy-First AI Prototype Building"
BUILDING_TYPE = "Generic office/university-style building"
PROJECT_CONTEXT = "Prototype context for Centre for Technology and Infusion, La Trobe University"
LAYOUT_NOTE = "Same abstract zone layout is reused across Level 1 and Level 2."
INFERENCE_NOTE = (
    "Evacuated, trapped, and notification outcomes are inferred from aggregate occupancy, "
    "route status, alarms, and event-log text. No person-level tracking is used."
)

ZONES = [
    Zone("Z1", "Reception", 10),
    Zone("Z2", "Workspace", 30),
    Zone("Z3", "MeetingRoom", 12),
    Zone("Z4", "Pantry", 8),
    Zone("Z5", "Corridor", 15),
    Zone("Z6", "ExitArea", 5),
]

ZONE_FUNCTIONS = {
    "Z1": "Front-of-house reception and entry area",
    "Z2": "Primary staff workspace or open office area",
    "Z3": "Enclosed meeting or collaboration room",
    "Z4": "Pantry or kitchen break area",
    "Z5": "Main corridor or circulation path",
    "Z6": "Exit or evacuation holding area",
}

HAZARD_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
NOTIFICATION_SIGNALS = [
    ("Wardens coordinating staged evacuation.", "Wardens coordinated staged evacuation"),
    ("Fire response team notified.", "Fire response team was notified"),
    ("Control room monitoring smoke migration.", "Control room monitored smoke migration"),
    ("Occupants directed toward nearest safe exit.", "Occupants were directed toward the nearest safe exit"),
]

SCENARIOS = [
    Scenario(
        incident_id="MON-20260401",
        building_level="Level 1",
        start_time=datetime(2026, 4, 1, 8, 0, 0),
        end_time=datetime(2026, 4, 1, 13, 55, 0),
        fire_start_time=None,
        fire_origin_zone=None,
        adjacent_zones=set(),
        event_family="Routine",
    ),
    Scenario(
        incident_id="INC-20260401-F1",
        building_level="Level 1",
        start_time=datetime(2026, 4, 1, 14, 0, 0),
        end_time=datetime(2026, 4, 1, 15, 0, 0),
        fire_start_time=datetime(2026, 4, 1, 14, 15, 0),
        fire_origin_zone="Z3",
        adjacent_zones={"Z2", "Z4", "Z5"},
        event_family="Fire",
    ),
    Scenario(
        incident_id="MON-20260402",
        building_level="Level 1",
        start_time=datetime(2026, 4, 2, 8, 0, 0),
        end_time=datetime(2026, 4, 2, 13, 55, 0),
        fire_start_time=None,
        fire_origin_zone=None,
        adjacent_zones=set(),
        event_family="Routine",
    ),
    Scenario(
        incident_id="INC-20260402-F2",
        building_level="Level 2",
        start_time=datetime(2026, 4, 2, 14, 0, 0),
        end_time=datetime(2026, 4, 2, 15, 0, 0),
        fire_start_time=datetime(2026, 4, 2, 14, 20, 0),
        fire_origin_zone="Z5",
        adjacent_zones={"Z2", "Z4", "Z6"},
        event_family="Fire",
    ),
]


def generate_timestamps(start: datetime, end: datetime, step_minutes: int) -> list[datetime]:
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=step_minutes)
    return timestamps


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def occupancy_profile(ts: datetime, zone: Zone, scenario: Scenario) -> int:
    hour = ts.hour + ts.minute / 60

    if 8 <= hour < 10:
        factor = 0.65 + (hour - 8) * 0.22
    elif 10 <= hour < 12:
        factor = 1.0
    elif 12 <= hour < 13:
        factor = 0.82
    elif 13 <= hour < 14:
        factor = 0.92
    else:
        factor = 0.88

    if zone.zone_id == "Z4" and 12 <= hour < 13:
        factor += 0.22
    if zone.zone_id == "Z6":
        factor *= 0.35
    if scenario.building_level == "Level 2" and zone.zone_id in {"Z2", "Z5"}:
        factor += 0.08

    baseline = max(0, int(round(zone.base_occ * factor + np.random.normal(0, 1.2))))

    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return baseline

    minutes_after = int((ts - scenario.fire_start_time).total_seconds() / 60)

    if zone.zone_id == scenario.fire_origin_zone:
        decay = max(0.03, 1 - minutes_after / 34)
    elif zone.zone_id in scenario.adjacent_zones:
        decay = max(0.02, 1 - minutes_after / 26)
    else:
        decay = max(0.0, 1 - minutes_after / 20)

    return max(0, int(round(baseline * decay + np.random.normal(0, 0.9))))


def split_population(total: int) -> tuple[int, int, int, int]:
    if total <= 0:
        return 0, 0, 0, 0

    child_count = np.random.binomial(total, 0.05)
    elderly_count = np.random.binomial(max(total - child_count, 0), 0.1)
    disabled_count = np.random.binomial(max(total - child_count - elderly_count, 0), 0.07)

    allocated = child_count + elderly_count + disabled_count
    adult_count = max(0, total - allocated)
    return adult_count, child_count, elderly_count, disabled_count


def fire_impact_multiplier(ts: datetime, zone: Zone, scenario: Scenario) -> float:
    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return 0.0

    minutes_after = int((ts - scenario.fire_start_time).total_seconds() / 60)

    if zone.zone_id == scenario.fire_origin_zone:
        rise = smoothstep(minutes_after / 18)
        pulse = 0.08 * np.sin(minutes_after / 4)
        return clamp(rise + pulse, 0.0, 1.0)

    if zone.zone_id in scenario.adjacent_zones:
        delayed = max(0, minutes_after - 5)
        rise = smoothstep(delayed / 24)
        pulse = 0.05 * np.sin(delayed / 5)
        return clamp(0.58 * rise + pulse, 0.0, 0.72)

    delayed = max(0, minutes_after - 10)
    rise = smoothstep(delayed / 28)
    pulse = 0.03 * np.sin(delayed / 6)
    return clamp(0.22 * rise + pulse, 0.0, 0.3)


def generate_environment(ts: datetime, zone: Zone, scenario: Scenario, prev_env: dict) -> dict:
    temperature_target = np.random.normal(23, 0.8)
    humidity_target = np.random.normal(46, 3.2)
    co2_target = np.random.normal(560, 45)
    voc_target = np.random.normal(105, 20)
    pm25_target = np.random.normal(9, 2.5)

    impact = fire_impact_multiplier(ts, zone, scenario)

    if impact > 0:
        temperature_target += impact * np.random.uniform(10, 30)
        humidity_target -= impact * np.random.uniform(4, 18)
        co2_target += impact * np.random.uniform(180, 1550)
        voc_target += impact * np.random.uniform(140, 1350)
        pm25_target += impact * np.random.uniform(45, 420)

    persistence = {
        "PM2.5": 0.72,
        "VOC": 0.7,
        "Temperature": 0.78,
        "CO2": 0.74,
        "Humidity": 0.7,
    }

    jitter_scale = 1 + (impact * 1.2)
    pm25 = (
        persistence["PM2.5"] * prev_env["PM2.5"]
        + (1 - persistence["PM2.5"]) * pm25_target
        + np.random.normal(0, 1.8 * jitter_scale)
    )
    voc = (
        persistence["VOC"] * prev_env["VOC"]
        + (1 - persistence["VOC"]) * voc_target
        + np.random.normal(0, 10 * jitter_scale)
    )
    temperature = (
        persistence["Temperature"] * prev_env["Temperature"]
        + (1 - persistence["Temperature"]) * temperature_target
        + np.random.normal(0, 0.45 * jitter_scale)
    )
    co2 = (
        persistence["CO2"] * prev_env["CO2"]
        + (1 - persistence["CO2"]) * co2_target
        + np.random.normal(0, 20 * jitter_scale)
    )
    humidity = (
        persistence["Humidity"] * prev_env["Humidity"]
        + (1 - persistence["Humidity"]) * humidity_target
        + np.random.normal(0, 1.6 * jitter_scale)
    )

    if 0 < impact < 0.3:
        pm25 += np.random.uniform(-4, 7)
        voc += np.random.uniform(-12, 18)
        co2 += np.random.uniform(-30, 55)

    return {
        "PM2.5": round(max(0, pm25), 2),
        "VOC": round(max(0, voc), 2),
        "Temperature": round(max(0, temperature), 2),
        "CO2": round(max(350, co2), 2),
        "Humidity": round(float(np.clip(humidity, 10, 95)), 2),
    }


def calculate_fire_hazard(
    pm25: float,
    voc: float,
    temperature: float,
    co2: float,
    humidity: float,
    prev_pm25: float,
    prev_voc: float,
    prev_temperature: float,
    prev_co2: float,
    prev_hazard_score: float,
    prev_hazard_rating: str,
) -> tuple[float, str]:
    pm_abs = clamp01((pm25 - 18) / 170)
    voc_abs = clamp01((voc - 140) / 560)
    temp_abs = clamp01((temperature - 28) / 18)
    co2_abs = clamp01((co2 - 700) / 900)
    humidity_risk = clamp01((35 - humidity) / 18)

    pm_rise = clamp01((pm25 - prev_pm25) / 70)
    voc_rise = clamp01((voc - prev_voc) / 220)
    temp_rise = clamp01((temperature - prev_temperature) / 7)
    co2_rise = clamp01((co2 - prev_co2) / 260)

    score = 100 * (
        0.28 * pm_abs
        + 0.18 * temp_abs
        + 0.18 * voc_abs
        + 0.12 * co2_abs
        + 0.06 * humidity_risk
        + 0.08 * pm_rise
        + 0.04 * temp_rise
        + 0.04 * voc_rise
        + 0.02 * co2_rise
    )

    if pm_abs > 0.55 and temp_abs > 0.35:
        score += 8
    if pm_abs > 0.55 and voc_abs > 0.35:
        score += 8
    if co2_abs > 0.45 and humidity_risk > 0.25:
        score += 4

    score += np.random.normal(0, 1.1)
    score = (0.72 * score) + (0.28 * prev_hazard_score)
    score = clamp(score, 0, 100)

    threshold_windows = [
        (20, "Low", "Moderate"),
        (40, "Moderate", "High"),
        (70, "High", "Critical"),
    ]

    if score < 20:
        rating = "Low"
    elif score < 40:
        rating = "Moderate"
    elif score < 70:
        rating = "High"
    else:
        rating = "Critical"

    for threshold, lower_rating, higher_rating in threshold_windows:
        if abs(score - threshold) <= 3:
            if prev_hazard_rating in {lower_rating, higher_rating}:
                rating = prev_hazard_rating
            elif score < threshold:
                rating = np.random.choice([lower_rating, higher_rating], p=[0.75, 0.25])
            else:
                rating = np.random.choice([higher_rating, lower_rating], p=[0.75, 0.25])
            break

    return round(score, 2), rating


def fire_alarm_triggered(
    ts: datetime,
    zone: Zone,
    scenario: Scenario,
    hazard_score: float,
    smoke_obscuration_pct: float,
) -> int:
    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return 0
    if zone.zone_id == scenario.fire_origin_zone:
        return int(hazard_score >= 35 or smoke_obscuration_pct >= 20)
    if zone.zone_id in scenario.adjacent_zones:
        return int(hazard_score >= 45 or smoke_obscuration_pct >= 25)
    return int(hazard_score >= 60 or smoke_obscuration_pct >= 35)


def smoke_obscuration_pct(pm25: float) -> float:
    return round(clamp01(pm25 / 260) * 100, 2)


def sprinkler_status(ts: datetime, zone: Zone, scenario: Scenario, hazard_rating: str) -> str:
    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return "Off"
    if zone.zone_id == scenario.fire_origin_zone and hazard_rating in {"High", "Critical"}:
        return "Activated"
    if zone.zone_id in scenario.adjacent_zones and hazard_rating == "Critical":
        return "Activated"
    return "Standby"


def zone_clearance_status(total_occ: int, ts: datetime, scenario: Scenario) -> str:
    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return "Not Started"
    if total_occ == 0:
        return "Cleared"
    if total_occ <= 3:
        return "Nearly Cleared"
    return "Occupied"


def exit_route_passable(zone: Zone, scenario: Scenario, hazard_rating: str) -> str:
    if zone.zone_id == scenario.fire_origin_zone and hazard_rating in {"High", "Critical"}:
        return "No"
    if zone.zone_id in scenario.adjacent_zones and hazard_rating == "Critical":
        return "Partial"
    return "Yes"


def evacuation_priority_score(
    elderly: int,
    disabled: int,
    children: int,
    total_occ: int,
    hazard_score: float,
) -> float:
    if total_occ <= 0:
        return 0.0

    vulnerable_weight = (children * 1.3) + (elderly * 1.5) + (disabled * 1.9)
    occupancy_factor = min(total_occ / 35, 1.0) * 20
    vulnerability_factor = min(vulnerable_weight / 10, 1.0) * 35
    hazard_factor = min(hazard_score / 100, 1.0) * 45
    return round(min(100, occupancy_factor + vulnerability_factor + hazard_factor), 2)


def last_manual_sweep_time(ts: datetime, clearance_status: str, scenario: Scenario):
    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return None
    if clearance_status in {"Cleared", "Nearly Cleared"}:
        minutes_back = int(np.random.choice([2, 3, 5, 7]))
        return (ts - timedelta(minutes=minutes_back)).strftime("%Y-%m-%d %H:%M:%S")
    return None


def event_type(ts: datetime, zone: Zone, scenario: Scenario, hazard_rating: str) -> str:
    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return "Routine Monitoring"
    if zone.zone_id == scenario.fire_origin_zone:
        return "Fire Emergency"
    if hazard_rating in {"High", "Critical"}:
        return "Smoke Spread Alert"
    return "Evacuation Update"


def incident_id_for_event(scenario: Scenario, row_event_type: str) -> str:
    if scenario.event_family == "Routine":
        return scenario.incident_id

    event_suffix = {
        "Routine Monitoring": "RM",
        "Evacuation Update": "EV",
        "Smoke Spread Alert": "SS",
        "Fire Emergency": "FE",
    }[row_event_type]
    return f"{scenario.incident_id}-{event_suffix}"


def build_details(
    ts: datetime,
    zone: Zone,
    total_occ: int,
    hazard_rating: str,
    fire_alarm: int,
    sprinkler: str,
    clearance: str,
    route_passable: str,
    scenario: Scenario,
) -> str:
    zone_label = f"{zone.zone_id} ({zone.zone_name})"

    if not scenario.fire_start_time or ts < scenario.fire_start_time:
        return (
            f"{zone_label} under normal monitoring. Occupancy at {total_occ} "
            f"with no active emergency indicators."
        )

    incident_note = fake.random_element([
        "Fire response team notified.",
        "Occupants directed toward nearest safe exit.",
        "Control room monitoring smoke migration.",
        "Wardens coordinating staged evacuation.",
    ])

    return (
        f"{zone_label} status: hazard {hazard_rating}, alarm "
        f"{'triggered' if fire_alarm else 'not triggered'}, sprinkler {sprinkler}, "
        f"clearance {clearance}, exit route {route_passable}. {incident_note}"
    )


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    previous_values = {
        (scenario.incident_id, zone.zone_id): {
            "PM2.5": 10.0,
            "VOC": 110.0,
            "Temperature": 23.0,
            "CO2": 550.0,
            "Humidity": 46.0,
            "Hazard_Level_Score": 1.5,
            "Hazard_Level_Rating": "Low",
        }
        for scenario in SCENARIOS
        for zone in ZONES
    }

    for scenario in SCENARIOS:
        timestamps = generate_timestamps(scenario.start_time, scenario.end_time, TIME_STEP_MINUTES)

        for ts in timestamps:
            for zone in ZONES:
                total_raw = occupancy_profile(ts, zone, scenario)
                adult_count, child_count, elderly_count, disabled_count = split_population(total_raw)
                prev = previous_values[(scenario.incident_id, zone.zone_id)]
                env = generate_environment(ts, zone, scenario, prev)
                hazard_score, hazard_rating = calculate_fire_hazard(
                    pm25=env["PM2.5"],
                    voc=env["VOC"],
                    temperature=env["Temperature"],
                    co2=env["CO2"],
                    humidity=env["Humidity"],
                    prev_pm25=prev["PM2.5"],
                    prev_voc=prev["VOC"],
                    prev_temperature=prev["Temperature"],
                    prev_co2=prev["CO2"],
                    prev_hazard_score=prev["Hazard_Level_Score"],
                    prev_hazard_rating=prev["Hazard_Level_Rating"],
                )

                smoke_pct = smoke_obscuration_pct(env["PM2.5"])
                fire_alarm = fire_alarm_triggered(ts, zone, scenario, hazard_score, smoke_pct)
                sprinkler = sprinkler_status(ts, zone, scenario, hazard_rating)
                clearance = zone_clearance_status(total_raw, ts, scenario)
                route_passable = exit_route_passable(zone, scenario, hazard_rating)
                evac_priority = evacuation_priority_score(
                    elderly=elderly_count,
                    disabled=disabled_count,
                    children=child_count,
                    total_occ=total_raw,
                    hazard_score=hazard_score,
                )
                sweep_time = last_manual_sweep_time(ts, clearance, scenario)
                row_event_type = event_type(ts, zone, scenario, hazard_rating)
                row_incident_id = incident_id_for_event(scenario, row_event_type)
                details = build_details(
                    ts=ts,
                    zone=zone,
                    total_occ=total_raw,
                    hazard_rating=hazard_rating,
                    fire_alarm=fire_alarm,
                    sprinkler=sprinkler,
                    clearance=clearance,
                    route_passable=route_passable,
                    scenario=scenario,
                )

                rows.append({
                    "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "Incident_ID": row_incident_id,
                    "Building_Level": scenario.building_level,
                    "Zone_ID": zone.zone_id,
                    "Zone_Name": zone.zone_name,
                    "Adult_Count": adult_count,
                    "Child_Count": child_count,
                    "Elderly_Count": elderly_count,
                    "Disabled_Count": disabled_count,
                    "Total_Occupancy_Raw": total_raw,
                    "Fire_Alarm_Triggered": fire_alarm,
                    "Smoke_Obscuration_Pct": smoke_pct,
                    "Sprinkler_System_Status": sprinkler,
                    "Hazard_Level_Rating": hazard_rating,
                    "Zone_Clearance_Status": clearance,
                    "Exit_Route_Passable": route_passable,
                    "Evacuation_Priority_Score": evac_priority,
                    "Last_Manual_Sweep_Time": sweep_time,
                    "Event_Type": row_event_type,
                    "Details": details,
                    "PM2.5": env["PM2.5"],
                    "VOC": env["VOC"],
                    "Temperature": env["Temperature"],
                    "CO2": env["CO2"],
                    "Humidity": env["Humidity"],
                    "Hazard_Level_Score": hazard_score,
                })

                previous_values[(scenario.incident_id, zone.zone_id)] = {
                    "PM2.5": env["PM2.5"],
                    "VOC": env["VOC"],
                    "Temperature": env["Temperature"],
                    "CO2": env["CO2"],
                    "Humidity": env["Humidity"],
                    "Hazard_Level_Score": hazard_score,
                    "Hazard_Level_Rating": hazard_rating,
                }

    df = pd.DataFrame(rows)

    final_columns = [
        "Timestamp",
        "Incident_ID",
        "Building_Level",
        "Zone_ID",
        "Zone_Name",
        "Adult_Count",
        "Child_Count",
        "Elderly_Count",
        "Disabled_Count",
        "Total_Occupancy_Raw",
        "Fire_Alarm_Triggered",
        "Smoke_Obscuration_Pct",
        "Sprinkler_System_Status",
        "Hazard_Level_Rating",
        "Zone_Clearance_Status",
        "Exit_Route_Passable",
        "Evacuation_Priority_Score",
        "Last_Manual_Sweep_Time",
        "Event_Type",
        "Details",
    ]

    enriched_columns = final_columns + [
        "PM2.5",
        "VOC",
        "Temperature",
        "CO2",
        "Humidity",
        "Hazard_Level_Score",
    ]

    return df[final_columns], df[enriched_columns]


def generate_zone_reference() -> pd.DataFrame:
    building_levels = list(dict.fromkeys(scenario.building_level for scenario in SCENARIOS))
    rows = []

    for level in building_levels:
        for zone in ZONES:
            rows.append({
                "Building_Name": BUILDING_NAME,
                "Building_Type": BUILDING_TYPE,
                "Is_Real_Building": "No",
                "Project_Context": PROJECT_CONTEXT,
                "Building_Level": level,
                "Zone_ID": zone.zone_id,
                "Zone_Name": zone.zone_name,
                "Zone_Function": ZONE_FUNCTIONS[zone.zone_id],
                "Layout_Note": LAYOUT_NOTE,
            })

    return pd.DataFrame(rows)


def ordered_unique(values) -> list[str]:
    seen = set()
    ordered = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return ordered


def format_timestamp(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def format_zone_label(zone_id: str, zone_name: str) -> str:
    return f"{zone_id} ({zone_name})"


def scenario_rows(df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    if scenario.event_family == "Routine":
        incident_df = df.loc[df["Incident_ID"] == scenario.incident_id].copy()
    else:
        incident_df = df.loc[df["Incident_ID"].astype(str).str.startswith(f"{scenario.incident_id}-")].copy()

    incident_df["Timestamp"] = pd.to_datetime(incident_df["Timestamp"])
    return incident_df.sort_values(["Timestamp", "Zone_ID"]).reset_index(drop=True)


def occupancy_at_timestamp(timeline: pd.DataFrame, ts: datetime | None) -> int:
    if ts is None:
        return 0

    match = timeline.loc[timeline["Timestamp"] == pd.Timestamp(ts), "Total_Occupancy"]
    if match.empty:
        return 0
    return int(match.iloc[0])


def summarize_notification_actions(incident_df: pd.DataFrame) -> list[str]:
    actions = []

    if incident_df["Fire_Alarm_Triggered"].eq(1).any():
        actions.append("Alarm system triggered")

    details = incident_df["Details"].fillna("")
    for text, label in NOTIFICATION_SIGNALS:
        if details.str.contains(text, regex=False).any():
            actions.append(label)

    return actions


def build_outcome_summary(
    scenario: Scenario,
    final_remaining_occupancy: int,
    final_blocked_count: int,
    final_impaired_count: int,
    final_rows: pd.DataFrame,
) -> str:
    final_cleared = int(final_rows["Zone_Clearance_Status"].eq("Cleared").sum())
    total_zones = len(final_rows)

    if scenario.event_family == "Routine":
        return "Routine monitoring period completed with no active emergency incident."

    if final_remaining_occupancy == 0 and final_cleared == total_zones:
        return "Incident concluded with all zones cleared and no remaining occupants."

    if final_blocked_count > 0:
        return "Incident concluded with occupants remaining in zones where the exit route was blocked."

    if final_impaired_count > 0:
        return "Incident concluded with occupants remaining in zones where the exit route was impaired."

    return "Incident concluded with some occupants still present, but all recorded exit routes remained passable."


def generate_incident_reports(
    enriched_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    summary_rows = []
    final_zone_rows = []

    for scenario in SCENARIOS:
        incident_df = scenario_rows(enriched_df, scenario)
        if incident_df.empty:
            continue

        timeline = (
            incident_df.groupby("Timestamp", as_index=False)
            .agg(Total_Occupancy=("Total_Occupancy_Raw", "sum"))
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
        timeline["Blocked_Route_Occupancy"] = [
            int(
                incident_df.loc[
                    (incident_df["Timestamp"] == ts) & (incident_df["Exit_Route_Passable"] == "No"),
                    "Total_Occupancy_Raw",
                ].sum()
            )
            for ts in timeline["Timestamp"]
        ]
        timeline["Impaired_Route_Occupancy"] = [
            int(
                incident_df.loc[
                    (incident_df["Timestamp"] == ts)
                    & (incident_df["Exit_Route_Passable"].isin(["No", "Partial"])),
                    "Total_Occupancy_Raw",
                ].sum()
            )
            for ts in timeline["Timestamp"]
        ]

        start_ts = timeline.iloc[0]["Timestamp"]
        end_ts = timeline.iloc[-1]["Timestamp"]
        duration_minutes = int((end_ts - start_ts).total_seconds() // 60)
        start_occupancy = int(timeline.iloc[0]["Total_Occupancy"])
        peak_timeline_row = timeline.sort_values(
            ["Total_Occupancy", "Timestamp"], ascending=[False, True]
        ).iloc[0]
        peak_total_occupancy = int(peak_timeline_row["Total_Occupancy"])
        peak_total_occupancy_time = peak_timeline_row["Timestamp"]
        fire_start_occupancy = occupancy_at_timestamp(timeline, scenario.fire_start_time)
        final_remaining_occupancy = int(timeline.iloc[-1]["Total_Occupancy"])
        evacuation_baseline = fire_start_occupancy if scenario.fire_start_time else start_occupancy
        estimated_evacuated = max(0, evacuation_baseline - final_remaining_occupancy)

        peak_blocked_row = timeline.sort_values(
            ["Blocked_Route_Occupancy", "Timestamp"], ascending=[False, True]
        ).iloc[0]
        peak_impaired_row = timeline.sort_values(
            ["Impaired_Route_Occupancy", "Timestamp"], ascending=[False, True]
        ).iloc[0]

        alarm_rows = incident_df.loc[incident_df["Fire_Alarm_Triggered"] == 1]
        first_alarm_time = alarm_rows["Timestamp"].min() if not alarm_rows.empty else None
        alarm_zones = ordered_unique(
            alarm_rows.apply(lambda row: format_zone_label(row["Zone_ID"], row["Zone_Name"]), axis=1).tolist()
        )

        sprinkler_rows = incident_df.loc[incident_df["Sprinkler_System_Status"] == "Activated"]
        first_sprinkler_time = (
            sprinkler_rows["Timestamp"].min() if not sprinkler_rows.empty else None
        )
        sprinkler_zones = ordered_unique(
            sprinkler_rows.apply(lambda row: format_zone_label(row["Zone_ID"], row["Zone_Name"]), axis=1).tolist()
        )

        hazard_ranked = incident_df.assign(
            Hazard_Rank=incident_df["Hazard_Level_Rating"].map(HAZARD_ORDER)
        )
        peak_hazard_row = hazard_ranked.sort_values(
            ["Hazard_Rank", "Hazard_Level_Score", "Timestamp"],
            ascending=[False, False, True],
        ).iloc[0]

        smoke_peak_row = incident_df.sort_values(
            ["Smoke_Obscuration_Pct", "Timestamp"], ascending=[False, True]
        ).iloc[0]

        final_rows = incident_df.loc[incident_df["Timestamp"] == end_ts].copy()
        final_rows = final_rows.sort_values("Zone_ID").reset_index(drop=True)
        final_blocked_count = int(
            final_rows.loc[final_rows["Exit_Route_Passable"] == "No", "Total_Occupancy_Raw"].sum()
        )
        final_impaired_count = int(
            final_rows.loc[
                final_rows["Exit_Route_Passable"].isin(["No", "Partial"]),
                "Total_Occupancy_Raw",
            ].sum()
        )
        final_cleared_count = int(final_rows["Zone_Clearance_Status"].eq("Cleared").sum())
        final_occupied_count = int(final_rows["Zone_Clearance_Status"].eq("Occupied").sum())
        final_nearly_cleared_count = int(
            final_rows["Zone_Clearance_Status"].eq("Nearly Cleared").sum()
        )
        final_impaired_zones = ordered_unique(
            final_rows.loc[
                final_rows["Exit_Route_Passable"].isin(["No", "Partial"])
            ].apply(lambda row: format_zone_label(row["Zone_ID"], row["Zone_Name"]), axis=1).tolist()
        )

        notification_actions = summarize_notification_actions(incident_df)
        outcome_summary = build_outcome_summary(
            scenario,
            final_remaining_occupancy,
            final_blocked_count,
            final_impaired_count,
            final_rows,
        )

        summary_rows.append({
            "Base_Incident_ID": scenario.incident_id,
            "Incident_Type": scenario.event_family,
            "Building_Name": BUILDING_NAME,
            "Building_Level": scenario.building_level,
            "Start_Time": format_timestamp(start_ts),
            "End_Time": format_timestamp(end_ts),
            "Duration_Minutes": duration_minutes,
            "Fire_Start_Time": format_timestamp(scenario.fire_start_time),
            "Fire_Origin_Zone": scenario.fire_origin_zone or "",
            "Fire_Origin_Zone_Name": (
                next((zone.zone_name for zone in ZONES if zone.zone_id == scenario.fire_origin_zone), "")
                if scenario.fire_origin_zone
                else ""
            ),
            "Peak_Total_Occupancy": peak_total_occupancy,
            "Peak_Total_Occupancy_Time": format_timestamp(peak_total_occupancy_time),
            "Occupancy_At_Incident_Start": start_occupancy,
            "Occupancy_At_Fire_Start": fire_start_occupancy,
            "Final_Remaining_Occupancy": final_remaining_occupancy,
            "Estimated_Evacuated_Count": estimated_evacuated,
            "Estimated_Trapped_Count_Final": final_blocked_count,
            "Peak_Trapped_Count": int(peak_blocked_row["Blocked_Route_Occupancy"]),
            "Peak_Trapped_Time": format_timestamp(peak_blocked_row["Timestamp"]),
            "Peak_Impaired_Route_Count": int(peak_impaired_row["Impaired_Route_Occupancy"]),
            "Peak_Impaired_Route_Time": format_timestamp(peak_impaired_row["Timestamp"]),
            "First_Alarm_Time": format_timestamp(first_alarm_time),
            "Alarm_Triggered_Zones": "; ".join(alarm_zones),
            "First_Sprinkler_Activation_Time": format_timestamp(first_sprinkler_time),
            "Sprinkler_Activated_Zones": "; ".join(sprinkler_zones),
            "Highest_Hazard_Rating": peak_hazard_row["Hazard_Level_Rating"],
            "Highest_Hazard_Score": round(float(peak_hazard_row["Hazard_Level_Score"]), 2),
            "Highest_Hazard_Time": format_timestamp(peak_hazard_row["Timestamp"]),
            "Highest_Hazard_Zone": format_zone_label(
                peak_hazard_row["Zone_ID"], peak_hazard_row["Zone_Name"]
            ),
            "Peak_Smoke_Obscuration_Pct": round(float(smoke_peak_row["Smoke_Obscuration_Pct"]), 2),
            "Peak_Smoke_Time": format_timestamp(smoke_peak_row["Timestamp"]),
            "Peak_Smoke_Zone": format_zone_label(
                smoke_peak_row["Zone_ID"], smoke_peak_row["Zone_Name"]
            ),
            "Final_Cleared_Zones_Count": final_cleared_count,
            "Final_Nearly_Cleared_Zones_Count": final_nearly_cleared_count,
            "Final_Occupied_Zones_Count": final_occupied_count,
            "Final_Impaired_Route_Zones": "; ".join(final_impaired_zones),
            "Notification_Actions_Observed": "; ".join(notification_actions),
            "Notification_Channel_Note": (
                "Explicit channels such as SMS, radio, mobile push, or PA announcements "
                "are not stored in the dataset."
            ),
            "Outcome_Summary": outcome_summary,
            "Inference_Note": INFERENCE_NOTE,
        })

        for _, row in final_rows.iterrows():
            final_zone_rows.append({
                "Base_Incident_ID": scenario.incident_id,
                "Incident_Type": scenario.event_family,
                "Building_Level": scenario.building_level,
                "Final_Timestamp": format_timestamp(end_ts),
                "Zone_ID": row["Zone_ID"],
                "Zone_Name": row["Zone_Name"],
                "Final_Occupancy": int(row["Total_Occupancy_Raw"]),
                "Final_Hazard_Level_Rating": row["Hazard_Level_Rating"],
                "Final_Hazard_Level_Score": round(float(row["Hazard_Level_Score"]), 2),
                "Final_Zone_Clearance_Status": row["Zone_Clearance_Status"],
                "Final_Exit_Route_Passable": row["Exit_Route_Passable"],
                "Final_Fire_Alarm_Triggered": int(row["Fire_Alarm_Triggered"]),
                "Final_Sprinkler_System_Status": row["Sprinkler_System_Status"],
                "Final_Smoke_Obscuration_Pct": round(float(row["Smoke_Obscuration_Pct"]), 2),
                "Final_Details": row["Details"],
            })

    summary_df = pd.DataFrame(summary_rows)
    final_zone_df = pd.DataFrame(final_zone_rows)

    lines = [
        "# Incident Summary Report",
        "",
        f"Building: {BUILDING_NAME}",
        f"Building type: {BUILDING_TYPE}",
        "Real building: No",
        "",
        "This report summarizes synthetic incident scenarios for the privacy-first emergency response prototype.",
        INFERENCE_NOTE,
        "",
    ]

    for _, summary in summary_df.iterrows():
        lines.extend([
            f"## {summary['Base_Incident_ID']} ({summary['Incident_Type']})",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Building level | {summary['Building_Level']} |",
            f"| Start time | {summary['Start_Time']} |",
            f"| End time | {summary['End_Time']} |",
            f"| Duration (minutes) | {summary['Duration_Minutes']} |",
            f"| Fire start time | {summary['Fire_Start_Time'] or 'Not applicable'} |",
            f"| Fire origin | {summary['Fire_Origin_Zone'] or 'Not applicable'} {f'({summary['Fire_Origin_Zone_Name']})' if summary['Fire_Origin_Zone_Name'] else ''} |",
            f"| Peak total occupancy | {summary['Peak_Total_Occupancy']} at {summary['Peak_Total_Occupancy_Time']} |",
            f"| Occupancy at incident start | {summary['Occupancy_At_Incident_Start']} |",
            f"| Occupancy at fire start | {summary['Occupancy_At_Fire_Start']} |",
            f"| Estimated evacuated count | {summary['Estimated_Evacuated_Count']} |",
            f"| Estimated trapped count at final state | {summary['Estimated_Trapped_Count_Final']} |",
            f"| Peak trapped count | {summary['Peak_Trapped_Count']} at {summary['Peak_Trapped_Time']} |",
            f"| Peak impaired-route count | {summary['Peak_Impaired_Route_Count']} at {summary['Peak_Impaired_Route_Time']} |",
            f"| Final remaining occupancy | {summary['Final_Remaining_Occupancy']} |",
            f"| Highest hazard | {summary['Highest_Hazard_Rating']} ({summary['Highest_Hazard_Score']}) at {summary['Highest_Hazard_Time']} in {summary['Highest_Hazard_Zone']} |",
            f"| Peak smoke obscuration | {summary['Peak_Smoke_Obscuration_Pct']} at {summary['Peak_Smoke_Time']} in {summary['Peak_Smoke_Zone']} |",
            f"| First alarm time | {summary['First_Alarm_Time'] or 'Not triggered'} |",
            f"| Alarmed zones | {summary['Alarm_Triggered_Zones'] or 'None recorded'} |",
            f"| First sprinkler activation | {summary['First_Sprinkler_Activation_Time'] or 'Not activated'} |",
            f"| Sprinkler activated zones | {summary['Sprinkler_Activated_Zones'] or 'None recorded'} |",
            f"| Final cleared zones | {summary['Final_Cleared_Zones_Count']} |",
            f"| Final nearly cleared zones | {summary['Final_Nearly_Cleared_Zones_Count']} |",
            f"| Final occupied zones | {summary['Final_Occupied_Zones_Count']} |",
            f"| Final impaired-route zones | {summary['Final_Impaired_Route_Zones'] or 'None'} |",
            "",
            "### Notification and Coordination",
            "",
            f"- Observed actions: {summary['Notification_Actions_Observed'] or 'No explicit coordination actions recorded'}",
            f"- Channel note: {summary['Notification_Channel_Note']}",
            "",
            "### Outcome",
            "",
            f"- {summary['Outcome_Summary']}",
            f"- {summary['Inference_Note']}",
            "",
            "### Final Zone State",
            "",
            "| Zone | Occupancy | Hazard | Clearance | Route | Alarm | Sprinkler |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])

        incident_final_rows = final_zone_df.loc[
            final_zone_df["Base_Incident_ID"] == summary["Base_Incident_ID"]
        ].sort_values("Zone_ID")
        for _, row in incident_final_rows.iterrows():
            lines.append(
                f"| {format_zone_label(row['Zone_ID'], row['Zone_Name'])} | "
                f"{row['Final_Occupancy']} | "
                f"{row['Final_Hazard_Level_Rating']} ({row['Final_Hazard_Level_Score']}) | "
                f"{row['Final_Zone_Clearance_Status']} | "
                f"{row['Final_Exit_Route_Passable']} | "
                f"{row['Final_Fire_Alarm_Triggered']} | "
                f"{row['Final_Sprinkler_System_Status']} |"
            )

        lines.append("")

    report_markdown = "\n".join(lines)
    return summary_df, final_zone_df, report_markdown


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Total rows: {len(df)}")


def save_text(content: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Text report saved to: {output_path}")


if __name__ == "__main__":
    final_dataset, enriched_dataset = generate_dataset()
    zone_reference = generate_zone_reference()
    incident_summary, final_zone_state, incident_report = generate_incident_reports(enriched_dataset)
    save_dataset(final_dataset, "data/emergency_zone_status.csv")
    save_dataset(enriched_dataset, "data/emergency_zone_status_enriched.csv")
    save_dataset(zone_reference, "data/building_zone_reference.csv")
    save_dataset(incident_summary, "data/incident_summary.csv")
    save_dataset(final_zone_state, "data/incident_final_zone_state.csv")
    save_text(incident_report, "data/incident_summary_report.md")
    print(final_dataset.head(12).to_string(index=False))
