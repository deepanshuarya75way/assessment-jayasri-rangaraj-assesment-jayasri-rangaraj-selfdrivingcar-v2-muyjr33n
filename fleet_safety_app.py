import os
import json
import sqlite3
import uuid
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
from ultralytics import YOLO
from typing import Any
# ---------------- CONFIGURATION ----------------

DB_PATH = "fleet_safety.db"
MODEL_PATH = "yolov8n.pt"
UPLOAD_FOLDER = Path("fleet_uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Object classes that may deserve human review.
# These detections do NOT prove dangerous driving.
REVIEW_CLASSES = {
    "person": ("Pedestrian visible", "Medium"),
    "bicycle": ("Cyclist visible", "Medium"),
    "motorcycle": ("Motorcycle visible", "Medium"),
    "traffic light": ("Traffic signal visible", "Low"),
    "stop sign": ("Stop sign visible", "Low"),
}

# ---------------- DATABASE ----------------

def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    with connect_db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_code TEXT UNIQUE NOT NULL,
            model TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_code TEXT UNIQUE NOT NULL,
            vehicle_id INTEGER NOT NULL,
            session_date TEXT NOT NULL,
            status TEXT NOT NULL,
            image_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(vehicle_id) REFERENCES vehicles(id)
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            box_json TEXT DEFAULT '[]',
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            description TEXT NOT NULL,
            evidence_count INTEGER DEFAULT 1,
            feedback TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_session_vehicle
        ON sessions(vehicle_id, session_date);

        CREATE INDEX IF NOT EXISTS idx_incident_status
        ON incidents(status, severity);
        """)

        # Add a sample vehicle only if no vehicles exist.
        count = con.execute(
            "SELECT COUNT(*) FROM vehicles"
        ).fetchone()[0]

        if count == 0:
            con.execute("""
                INSERT INTO vehicles
                (vehicle_code, model, notes, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                "DEMO-001",
                "Demo vehicle",
                "Sample vehicle",
                datetime.now().isoformat(timespec="seconds")
            ))


# ---------------- LOAD YOLO MODEL ----------------

model: Any = None
model_status: str = ""

try:
    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
        model_status = "YOLOv8n model loaded successfully."
    else:
        model_status = (
            "Model file not found. Put yolov8n.pt "
            "beside fleet_safety_app.py."
        )
except Exception as e:
    model_status = f"Model loading failed: {e}"


# ---------------- VEHICLE MANAGEMENT ----------------

def get_vehicles():
    with connect_db() as con:
        rows = con.execute("""
            SELECT id, vehicle_code, model
            FROM vehicles
            ORDER BY vehicle_code
        """).fetchall()

    return [
        f"{r['id']} | {r['vehicle_code']} | {r['model']}"
        for r in rows
    ]


def extract_vehicle_id(choice):
    if not choice:
        return None

    try:
        return int(choice.split("|")[0].strip())
    except (ValueError, AttributeError):
        return None


def add_vehicle(code, model, notes):
    code = (code or "").strip()

    if not code:
        return "Please enter a vehicle code.", gr.update()

    try:
        with connect_db() as con:
            con.execute("""
                INSERT INTO vehicles
                (vehicle_code, model, notes, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                code,
                (model or "").strip(),
                (notes or "").strip(),
                datetime.now().isoformat(timespec="seconds")
            ))

        choices = get_vehicles()
        selected = next(
            (v for v in choices if v.split("|")[1].strip() == code),
            None
        )

        return f"Vehicle {code} added successfully.", gr.update(
            choices=choices, value=selected
        )

    except sqlite3.IntegrityError:
        return "That vehicle code already exists.", gr.update(
            choices=get_vehicles()
        )


# ---------------- SESSION HISTORY ----------------

def session_history():
    with connect_db() as con:
        rows = con.execute("""
            SELECT
                s.session_code,
                v.vehicle_code,
                s.session_date,
                s.status,
                s.image_count,
                (SELECT COUNT(*) FROM detections d
                 WHERE d.session_id = s.id) AS detections,
                (SELECT COUNT(*) FROM incidents i
                 WHERE i.session_id = s.id) AS incidents
            FROM sessions s
            JOIN vehicles v ON v.id = s.vehicle_id
            ORDER BY s.id DESC
            LIMIT 100
        """).fetchall()

    if not rows:
        return "No sessions found."

    return "\n".join(
        f"{r['session_code']} | Vehicle: {r['vehicle_code']} | "
        f"Date: {r['session_date']} | Status: {r['status']} | "
        f"Images: {r['image_count']} | "
        f"Detections: {r['detections']} | "
        f"Incidents: {r['incidents']}"
        for r in rows
    )


# ---------------- IMAGE PROCESSING ----------------

def process_session(vehicle_choice, images, session_date):
    vehicle_id = extract_vehicle_id(vehicle_choice)

    if vehicle_id is None:
        return "Please select a vehicle.", session_history()

    if not images:
        return "Please upload at least one road image.", session_history()

    if model is None:
        return model_status, session_history()

    date_value = (session_date or "").strip()

    if not date_value:
        date_value = datetime.now().date().isoformat()

    try:
        datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError:
        return "Enter the date in YYYY-MM-DD format.", session_history()

    session_code = "SES-" + uuid.uuid4().hex[:8].upper()

    # Start a session record before processing.
    with connect_db() as con:
        cursor = con.execute("""
            INSERT INTO sessions
            (session_code, vehicle_id, session_date,
             status, image_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_code,
            vehicle_id,
            date_value,
            "processing",
            len(images),
            datetime.now().isoformat(timespec="seconds")
        ))

        session_id = cursor.lastrowid

    summaries = []
    total_detections = 0

    try:
        for uploaded_file in images:
            source = str(uploaded_file)

            original_name = Path(source).name
            saved_path = UPLOAD_FOLDER / (
                f"{session_code}_{uuid.uuid4().hex[:6]}_"
                f"{original_name}"
            )

            shutil.copy2(source, saved_path)

            # Use the existing YOLOv8n model.
            results = model.predict(
                source=str(saved_path),
                conf=0.25,
                verbose=False
            )

            result = results[0]
            image_count = 0

            with connect_db() as con:
                if result.boxes is not None:
                    for box in result.boxes:
                        box: Any
                        class_id = int(box.cls[0].item())
                        class_name = str(result.names[class_id])
                        confidence = float(box.conf[0].item())
                        coordinates = [
                            float(x)
                            for x in box.xyxy[0].tolist()
                        ]

                        con.execute("""
                            INSERT INTO detections
                            (session_id, image_name, class_name,
                             confidence, box_json)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            session_id,
                            original_name,
                            class_name,
                            confidence,
                            json.dumps(coordinates)
                        ))

                        image_count += 1
                        total_detections += 1

                        # Generate review flags for selected classes.
                        # These are NOT confirmed safety violations.
                        if (
                            class_name in REVIEW_CLASSES
                            and confidence >= 0.50
                        ):
                            category, severity = REVIEW_CLASSES[
                                class_name
                            ]

                            existing = con.execute("""
                                SELECT id FROM incidents
                                WHERE session_id = ?
                                  AND category = ?
                            """, (
                                session_id,
                                category
                            )).fetchone()

                            if existing:
                                con.execute("""
                                    UPDATE incidents
                                    SET evidence_count = evidence_count + 1
                                    WHERE id = ?
                                """, (existing["id"],))
                            else:
                                con.execute("""
                                    INSERT INTO incidents
                                    (session_id, title, category,
                                     severity, status, description,
                                     evidence_count, created_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    session_id,
                                    category,
                                    category,
                                    severity,
                                    "Open",
                                    (
                                        f"YOLO detected {class_name} "
                                        f"with confidence "
                                        f"{confidence:.2f} in "
                                        f"{original_name}. "
                                        "Human review is required. "
                                        "This does not prove unsafe "
                                        "driving or a collision."
                                    ),
                                    1,
                                    datetime.now().isoformat(
                                        timespec="seconds"
                                    )
                                ))

            summaries.append(
                f"{original_name}: {image_count} objects detected."
            )

        with connect_db() as con:
            con.execute("""
                UPDATE sessions SET status = 'complete'
                WHERE id = ?
            """, (session_id,))

        message = (
            f"Session {session_code} completed.\n"
            f"Images processed: {len(images)}\n"
            f"Total detections: {total_detections}\n\n"
            + "\n".join(summaries)
        )

        return message, session_history()

    except Exception as e:
        with connect_db() as con:
            con.execute("""
                UPDATE sessions SET status = 'incomplete'
                WHERE id = ?
            """, (session_id,))

        return (
            f"Session {session_code} was incomplete.\nError: {e}",
            session_history()
        )


# ---------------- INCIDENT REVIEW ----------------

def show_incidents(vehicle, severity, status, date_from, date_to):
    query = """
        SELECT i.*, s.session_code, s.session_date,
               v.vehicle_code
        FROM incidents i
        JOIN sessions s ON s.id = i.session_id
        JOIN vehicles v ON v.id = s.vehicle_id
        WHERE 1 = 1
    """

    params = []

    vehicle_id = extract_vehicle_id(vehicle)

    if vehicle_id is not None:
        query += " AND v.id = ?"
        params.append(vehicle_id)

    if severity != "All":
        query += " AND i.severity = ?"
        params.append(severity)

    if status != "All":
        query += " AND i.status = ?"
        params.append(status)

    if date_from:
        query += " AND s.session_date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND s.session_date <= ?"
        params.append(date_to)

    query += """
        ORDER BY
        CASE i.severity
            WHEN 'High' THEN 1
            WHEN 'Medium' THEN 2
            ELSE 3
        END,
        s.session_date DESC
    """

    with connect_db() as con:
        rows = con.execute(query, params).fetchall()

    if not rows:
        return "No incidents match these filters."

    output = []

    for r in rows:
        output.append(
            f"Incident ID: {r['id']}\n"
            f"Vehicle: {r['vehicle_code']}\n"
            f"Session: {r['session_code']}\n"
            f"Date: {r['session_date']}\n"
            f"Severity: {r['severity']}\n"
            f"Status: {r['status']}\n"
            f"Evidence count: {r['evidence_count']}\n"
            f"Title: {r['title']}\n"
            f"Details: {r['description']}\n"
            f"Manager feedback: {r['feedback'] or '(none)'}"
        )

    return "\n\n--------------------\n\n".join(output)


def update_incident(incident_id, status, feedback):
    try:
        incident_id = int(incident_id)
    except (ValueError, TypeError):
        return "Enter a valid incident ID."

    with connect_db() as con:
        cursor = con.execute("""
            UPDATE incidents
            SET status = ?, feedback = ?
            WHERE id = ?
        """, (
            status,
            (feedback or "").strip(),
            incident_id
        ))

    if cursor.rowcount == 0:
        return f"Incident {incident_id} was not found."

    return f"Incident {incident_id} updated successfully."


# ---------------- ANALYTICS ----------------

def fleet_analytics():
    with connect_db() as con:
        vehicles = con.execute(
            "SELECT COUNT(*) FROM vehicles"
        ).fetchone()[0]

        sessions = con.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]

        detections = con.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        incidents = con.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

        vehicle_rows = con.execute("""
            SELECT v.vehicle_code, COUNT(i.id) AS total
            FROM vehicles v
            LEFT JOIN sessions s ON s.vehicle_id = v.id
            LEFT JOIN incidents i ON i.session_id = s.id
            GROUP BY v.id
            ORDER BY total DESC
        """).fetchall()

        status_rows = con.execute("""
            SELECT status, COUNT(*) AS total
            FROM incidents
            GROUP BY status
        """).fetchall()

    text = (
        f"Total vehicles: {vehicles}\n"
        f"Total sessions: {sessions}\n"
        f"Total detections: {detections}\n"
        f"Total incidents: {incidents}\n\n"
        "Incidents by vehicle:\n"
    )

    text += "\n".join(
        f"{r['vehicle_code']}: {r['total']}"
        for r in vehicle_rows
    ) or "No vehicle data"

    text += "\n\nIncidents by status:\n"

    text += "\n".join(
        f"{r['status']}: {r['total']}"
        for r in status_rows
    ) or "No incidents yet"

    return text

# ---------------- GRADIO DASHBOARD ----------------

init_db()

with gr.Blocks(title="Fleet Safety Intelligence") as app:

    gr.Markdown(
        "# Fleet Safety Intelligence and Incident Management\n"
        "Manage vehicles, process road images, review incidents, "
        "and monitor fleet trends."
    )

    gr.Markdown(f"**Model status:** {model_status}")

    # VEHICLES TAB
    with gr.Tab("Vehicles"):
        gr.Markdown("Register each vehicle using a unique vehicle code.")

        vehicle_code = gr.Textbox(label="Vehicle code")
        vehicle_model = gr.Textbox(label="Vehicle make/model")
        vehicle_notes = gr.Textbox(label="Notes")

        add_button = gr.Button("Add vehicle")
        vehicle_message = gr.Textbox(label="Result")

        vehicle_list = gr.Dropdown(
            choices=get_vehicles(),
            label="Registered vehicles"
        )

        add_button.click(
            add_vehicle,
            inputs=[
                vehicle_code,
                vehicle_model,
                vehicle_notes
            ],
            outputs=[vehicle_message, vehicle_list]
        )

    # DRIVING SESSIONS TAB
    with gr.Tab("Driving Sessions"):
        session_vehicle = gr.Dropdown(
            choices=get_vehicles(),
            label="Select vehicle"
        )

        session_date = gr.Textbox(
            label="Session date (YYYY-MM-DD)",
            value=datetime.now().date().isoformat()
        )

        road_images = gr.File(
            label="Upload multiple road images",
            file_count="multiple",
            file_types=["image"],
            type="filepath"
        )

        process_button = gr.Button(
            "Process images",
            variant="primary"
        )

        process_result = gr.Textbox(
            label="Processing result",
            lines=10
        )

        history_result = gr.Textbox(
            label="Session history",
            lines=10
        )

        process_button.click(
            process_session,
            inputs=[
                session_vehicle,
                road_images,
                session_date
            ],
            outputs=[process_result, history_result]
        )

        history_button = gr.Button("Refresh session history")
        history_button.click(
            session_history,
            outputs=history_result
        )

    # INCIDENT REVIEW TAB
    with gr.Tab("Incident Review"):
        gr.Markdown(
            "Review flags require human verification. "
            "Object detection alone does not establish a violation."
        )

        filter_vehicle = gr.Dropdown(
            choices=["All"] + get_vehicles(),
            value="All",
            label="Vehicle filter"
        )

        filter_severity = gr.Dropdown(
            choices=["All", "High", "Medium", "Low"],
            value="All",
            label="Severity"
        )

        filter_status = gr.Dropdown(
            choices=["All", "Open", "In review", "Resolved"],
            value="All",
            label="Incident status"
        )

        filter_from = gr.Textbox(
            label="From date (optional, YYYY-MM-DD)"
        )

        filter_to = gr.Textbox(
            label="To date (optional, YYYY-MM-DD)"
        )

        incident_button = gr.Button("Show incidents")

        incident_output = gr.Textbox(
            label="Incident details and IDs",
            lines=18
        )

        incident_button.click(
            show_incidents,
            inputs=[
                filter_vehicle,
                filter_severity,
                filter_status,
                filter_from,
                filter_to
            ],
            outputs=incident_output
        )

        gr.Markdown("### Record a manager decision")

        incident_id_input = gr.Number(
            label="Incident ID",
            precision=0
        )

        new_status = gr.Dropdown(
            choices=["Open", "In review", "Resolved"],
            value="In review",
            label="New status"
        )

        manager_feedback = gr.Textbox(
            label="Manager feedback",
            lines=3
        )

        save_review = gr.Button("Save review")
        review_result = gr.Textbox(label="Update result")

        save_review.click(
            update_incident,
            inputs=[
                incident_id_input,
                new_status,
                manager_feedback
            ],
            outputs=review_result
        )

    # ANALYTICS TAB
    with gr.Tab("Analytics"):
        analytics_button = gr.Button("Refresh analytics")
        analytics_output = gr.Textbox(
            label="Fleet summary",
            lines=15,
            value=fleet_analytics()
        )

        analytics_button.click(
            fleet_analytics,
            outputs=analytics_output
        )


if __name__ == "__main__":
    app.launch()



