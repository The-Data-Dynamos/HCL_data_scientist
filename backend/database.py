import sqlite3
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

DB_PATH_ENV = os.environ.get("SMARTLOAN_DB_PATH", "smartloan.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    target_path = db_path or os.environ.get("SMARTLOAN_DB_PATH", DB_PATH_ENV)
    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the SQLite database schema if not already created."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assessment_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    loan_intent TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    contract_version TEXT NOT NULL,
                    decision_threshold REAL NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    probability_approved REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_assessment_created_at
                ON assessment_audits(created_at DESC);
            """)
    finally:
        conn.close()


def insert_assessment(
    loan_intent: str,
    model_name: str,
    model_version: str,
    contract_version: str,
    decision_threshold: float,
    predicted_class: int,
    probability_approved: float,
    created_at: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """
    Insert a non-sensitive assessment audit record.
    Applicant PII/income/credit score is strictly excluded from storage.
    """
    ts = created_at or datetime.now(timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO assessment_audits (
                    loan_intent, model_name, model_version, contract_version,
                    decision_threshold, predicted_class, probability_approved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    loan_intent,
                    model_name,
                    model_version,
                    contract_version,
                    decision_threshold,
                    predicted_class,
                    probability_approved,
                    ts
                )
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_assessments(limit: int = 100, offset: int = 0, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve audit records ordered by newest first."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, loan_intent, model_name, model_version, contract_version,
                   decision_threshold, predicted_class, probability_approved, created_at
            FROM assessment_audits
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_assessment_by_id(assessment_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single non-sensitive assessment audit record by ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, loan_intent, model_name, model_version, contract_version,
                   decision_threshold, predicted_class, probability_approved, created_at
            FROM assessment_audits
            WHERE id = ?
            """,
            (assessment_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
