import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "simulation.db"
TABLE_NAME = "incident_events"




def create_connection(db_path: str):
    return sqlite3.connect(db_path)



def create_table(conn: sqlite3.Connection):
    query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        "Timestamp" TEXT NOT NULL,
        "Incident_ID" TEXT NOT NULL,
        "Building_Level" INTEGER NOT NULL,
        "Zone_ID" TEXT NOT NULL,
        "Adult_Count" INTEGER NOT NULL,
        "Child_Count" INTEGER NOT NULL,
        "Elderly_Count" INTEGER NOT NULL,
        "Disabled_Count" INTEGER NOT NULL,
        "Total_Occupancy_Raw" INTEGER NOT NULL,
        "Fire_Alarm_Triggered" INTEGER NOT NULL,
        "Smoke_Obscuration_Pct" REAL NOT NULL,
        "Sprinkler_System_Status" TEXT NOT NULL,
        "Hazard_Level_Rating" REAL NOT NULL,
        "Zone_Clearance_Status" TEXT NOT NULL,
        "Exit_Route_Passable" INTEGER NOT NULL,
        "Evacuation_Priority_Score" REAL NOT NULL,
        "Last_Manual_Sweep_Time" TEXT NOT NULL,
        "PM2.5" REAL NOT NULL,
        "VOC" REAL NOT NULL,
        "Temperature" REAL NOT NULL,
        "CO2" REAL NOT NULL,
        "Humidity" REAL NOT NULL
    );
    """
    conn.execute(query)
    conn.commit()


def build_hourly_counts(total_rows=91, total_hours=24):
    """
    Match your sample shape:
    91 rows over 24 hours.
    This creates:
    first 19 hours -> 4 rows each
    last 5 hours  -> 3 rows each
    19*4 + 5*3 = 91
    """
    counts = [4] * total_hours
    extra = sum(counts) - total_rows

    idx = total_hours - 1
    while extra > 0:
        if counts[idx] > 1:
            counts[idx] -= 1
            extra -= 1
        idx -= 1

    return counts


def generate_incident_id(counter: int) -> str:
    return f"INC_{1000 + counter}"


def generate_zone_id() -> str:
    return f"Z{random.randint(1, 20)}"


def generate_last_manual_sweep_time() -> str:
    """
    Your sample looks like values such as:
    30:29.1, 59:29.1, 02:29.1
    So we mimic that format directly.
    """
    minute_part = random.randint(0, 59)
    return f"{minute_part:02d}:29.1"


def generate_row(ts: datetime, incident_counter: int) -> dict:
    building_level = random.randint(1, 10)
    zone_id = generate_zone_id()

    adult_count = random.randint(5, 40)
    child_count = random.randint(0, 10)
    elderly_count = random.randint(0, 8)
    disabled_count = random.randint(0, 5)
    total_occupancy = adult_count + child_count + elderly_count + disabled_count

    fire_alarm_triggered = random.choice([0, 1])
    sprinkler_status = random.choice(["ON", "OFF"])
    zone_clearance = random.choice(["Cleared", "Not Cleared"])
    exit_route_passable = random.choice([0, 1])

    smoke = round(random.uniform(2.0, 99.5), 2)
    pm25 = round(random.uniform(6.5, 146.5), 2)
    voc = round(random.uniform(0.2, 4.9), 2)
    temperature = round(random.uniform(18.5, 59.3), 2)
    co2 = round(random.uniform(300.0, 1996.0), 2)
    humidity = round(random.uniform(20.0, 89.9), 2)

    # Derived scores so the data feels related instead of fully random
    hazard = (
        (smoke / 100.0) * 4.0
        + (temperature / 60.0) * 1.6
        + (co2 / 2000.0) * 1.4
        + (1.0 if fire_alarm_triggered == 1 else 0.0) * 0.8
        + (0.5 if sprinkler_status == "OFF" else 0.0)
    )
    hazard = round(max(2.5, min(hazard, 8.8)), 2)

    evacuation_priority = (
        hazard * 0.6
        + total_occupancy * 0.03
        + (0.6 if exit_route_passable == 0 else 0.0)
        + (0.4 if zone_clearance == "Not Cleared" else 0.0)
    )
    evacuation_priority = round(max(2.0, min(evacuation_priority, 7.5)), 2)

    return {
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Incident_ID": generate_incident_id(incident_counter),
        "Building_Level": building_level,
        "Zone_ID": zone_id,
        "Adult_Count": adult_count,
        "Child_Count": child_count,
        "Elderly_Count": elderly_count,
        "Disabled_Count": disabled_count,
        "Total_Occupancy_Raw": total_occupancy,
        "Fire_Alarm_Triggered": fire_alarm_triggered,
        "Smoke_Obscuration_Pct": smoke,
        "Sprinkler_System_Status": sprinkler_status,
        "Hazard_Level_Rating": hazard,
        "Zone_Clearance_Status": zone_clearance,
        "Exit_Route_Passable": exit_route_passable,
        "Evacuation_Priority_Score": evacuation_priority,
        "Last_Manual_Sweep_Time": generate_last_manual_sweep_time(),
        "PM2.5": pm25,
        "VOC": voc,
        "Temperature": temperature,
        "CO2": co2,
        "Humidity": humidity,
    }


def insert_row(conn: sqlite3.Connection, row: dict):
    query = f"""
    INSERT INTO {TABLE_NAME} (
        "Timestamp",
        "Incident_ID",
        "Building_Level",
        "Zone_ID",
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
        "PM2.5",
        "VOC",
        "Temperature",
        "CO2",
        "Humidity"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        row["Timestamp"],
        row["Incident_ID"],
        row["Building_Level"],
        row["Zone_ID"],
        row["Adult_Count"],
        row["Child_Count"],
        row["Elderly_Count"],
        row["Disabled_Count"],
        row["Total_Occupancy_Raw"],
        row["Fire_Alarm_Triggered"],
        row["Smoke_Obscuration_Pct"],
        row["Sprinkler_System_Status"],
        row["Hazard_Level_Rating"],
        row["Zone_Clearance_Status"],
        row["Exit_Route_Passable"],
        row["Evacuation_Priority_Score"],
        row["Last_Manual_Sweep_Time"],
        row["PM2.5"],
        row["VOC"],
        row["Temperature"],
        row["CO2"],
        row["Humidity"],
    )

    conn.execute(query, values)


def generate_dataset_like_sample(
    start_time: datetime,
    total_rows: int = 91,
    total_hours: int = 24
):
    conn = create_connection(DB_PATH)

    try:
        create_table(conn)

        hourly_counts = build_hourly_counts(total_rows=total_rows, total_hours=total_hours)

        incident_counter = 1

        for hour_offset in range(total_hours):
            ts = start_time + timedelta(hours=hour_offset)

            rows_this_hour = hourly_counts[hour_offset]

            for _ in range(rows_this_hour):
                row = generate_row(ts, incident_counter)
                insert_row(conn, row)
                incident_counter += 1

        conn.commit()

        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {TABLE_NAME} LIMIT 10')
        rows = cursor.fetchall()

        print("Rows in table:")
        for row in rows:
            print(row)

        total = conn.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}').fetchone()[0]
        print("Synthetic data generated successfully.")
        print("Total rows in table:", total)

    except Exception as exc:
        conn.rollback()
        raise RuntimeError(f"Generation failed: {exc}") from exc

    finally:
        conn.close()





if __name__ == "__main__":
    # This start time mimics your sample pattern closely
    sample_start = datetime(2026, 4, 7, 0, 29, 6)

    generate_dataset_like_sample(
        start_time=sample_start,
        total_rows=91,
        total_hours=24
    )

