"""Phase D — SQLite WAL persistence layer and audit store.

Implements the single identity authority invariants (TechGAR.md Phase D):
- GID is a monotonic counter per site_id, stored in SQLite and never reused.
- All lifecycle transitions (mint, promote, handoff, park, unpark, alias, retire)
  are audited and durable.
- Kinematic checkpoints are stored periodically for post-crash recovery.
- Sessions and reservations use optimistic concurrency control via monotonic `revision`.
- Soft reset rebuilds tracker state in RAM but preserves SQLite state, GID sequence,
  parked identities, and active sessions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_SCHEMA_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: """
    -- 1. GID sequence counter per site
    CREATE TABLE IF NOT EXISTS identity_sequence (
        site_id TEXT PRIMARY KEY,
        last_global_id INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL
    );

    -- 2. Identities table (canonical GID, lifecycle state)
    CREATE TABLE IF NOT EXISTS identities (
        global_id INTEGER PRIMARY KEY,
        site_id TEXT NOT NULL,
        canonical_global_id INTEGER,
        lifecycle_state TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_observed_at REAL NOT NULL,
        slot_id TEXT,
        primary_camera TEXT DEFAULT '',
        origin_x REAL,
        origin_y REAL,
        max_displacement REAL DEFAULT 0.0,
        updated_at REAL NOT NULL
    );

    -- 3. Identity aliases (secondary -> canonical)
    CREATE TABLE IF NOT EXISTS identity_aliases (
        secondary_global_id INTEGER PRIMARY KEY,
        canonical_global_id INTEGER NOT NULL,
        aliased_at REAL NOT NULL,
        reason TEXT DEFAULT ''
    );

    -- 4. Append-only identity events audit log
    CREATE TABLE IF NOT EXISTS identity_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        frame_sequence INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        global_id INTEGER,
        camera_id TEXT DEFAULT '',
        detail TEXT DEFAULT '',
        evidence_json TEXT DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_events_gid ON identity_events (global_id);
    CREATE INDEX IF NOT EXISTS idx_events_ts ON identity_events (timestamp);

    -- 5. Kinematic checkpoints for crash recovery
    CREATE TABLE IF NOT EXISTS identity_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        global_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        pos_x REAL NOT NULL,
        pos_y REAL NOT NULL,
        vel_x REAL NOT NULL,
        vel_y REAL NOT NULL,
        cov_00 REAL NOT NULL,
        cov_01 REAL NOT NULL,
        cov_10 REAL NOT NULL,
        cov_11 REAL NOT NULL,
        slot_id TEXT,
        camera_id TEXT DEFAULT '',
        gallery_json TEXT DEFAULT '[]'
    );
    CREATE INDEX IF NOT EXISTS idx_chk_gid_ts ON identity_checkpoints (global_id, timestamp);

    -- 6. Sessions (QR/driver session lifecycle with optimistic locking)
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        site_id TEXT NOT NULL,
        global_vehicle_id INTEGER,
        state TEXT NOT NULL DEFAULT 'WAITING_FOR_SCAN',
        target_spot_id TEXT,
        parked_spot_id TEXT,
        qr_token_hash TEXT,
        qr_expires_at REAL,
        revision INTEGER NOT NULL DEFAULT 1,
        claimed_at REAL,
        updated_at REAL NOT NULL,
        parked_at REAL,
        exit_started_at REAL,
        closed_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_gid ON sessions (global_vehicle_id);

    -- 7. Slot reservations (atomic leases)
    CREATE TABLE IF NOT EXISTS reservations (
        reservation_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        global_vehicle_id INTEGER,
        slot_id TEXT NOT NULL UNIQUE,
        lease_expires_at REAL NOT NULL,
        state TEXT NOT NULL DEFAULT 'active',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );

    -- 8. Runtime epochs (audit trail of process starts and modes)
    CREATE TABLE IF NOT EXISTS runtime_epochs (
        epoch_id TEXT PRIMARY KEY,
        runtime_id TEXT NOT NULL,
        site_id TEXT NOT NULL,
        source_mode TEXT NOT NULL,
        started_at REAL NOT NULL,
        config_hash TEXT DEFAULT '',
        ended_at REAL
    );
    """,
}


class PersistenceStore:
    """Thread-safe SQLite WAL store for TechGAR identity, session and audit records."""

    def __init__(self, db_path: str | Path = ":memory:", site_id: str = "default_site") -> None:
        self.db_path = str(db_path)
        self.site_id = str(site_id)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None or self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA foreign_keys = ON;")
                self._conn.execute("PRAGMA busy_timeout = 5000;")
            return self._conn
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL,
                    description TEXT
                );
            """)
            conn.commit()
            cursor = conn.execute("SELECT MAX(version) FROM schema_migrations;")
            row = cursor.fetchone()
            current_v = row[0] if row and row[0] is not None else 0
            for version in sorted(MIGRATIONS):
                if version > current_v:
                    conn.executescript(MIGRATIONS[version])
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?);",
                        (version, time.time(), f"Migration {version}"),
                    )
                    conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- Identity Sequence & GID Allocation ----------------------------------

    def next_global_id(self, site_id: str | None = None) -> int:
        """Atomically increment and return the next monotonic GID for this site."""
        target_site = site_id or self.site_id
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            cursor = conn.execute(
                "SELECT last_global_id FROM identity_sequence WHERE site_id = ?;",
                (target_site,),
            )
            row = cursor.fetchone()
            if row is None:
                new_id = 1
                conn.execute(
                    "INSERT INTO identity_sequence (site_id, last_global_id, updated_at) VALUES (?, ?, ?);",
                    (target_site, new_id, now),
                )
            else:
                new_id = int(row["last_global_id"]) + 1
                conn.execute(
                    "UPDATE identity_sequence SET last_global_id = ?, updated_at = ? WHERE site_id = ?;",
                    (new_id, now, target_site),
                )
            conn.commit()
            return new_id

    def current_global_id(self, site_id: str | None = None) -> int:
        """Get current maximum allocated GID without incrementing."""
        target_site = site_id or self.site_id
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT last_global_id FROM identity_sequence WHERE site_id = ?;",
                (target_site,),
            )
            row = cursor.fetchone()
            return int(row["last_global_id"]) if row else 0

    # --- Identities CRUD -----------------------------------------------------

    def save_identity(self, global_id: int, lifecycle_state: str, created_at: float,
                      last_observed_at: float, slot_id: str | None = None,
                      primary_camera: str = "", origin_pos: tuple[float, float] | None = None,
                      max_displacement: float = 0.0, canonical_global_id: int | None = None) -> None:
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            orig_x = float(origin_pos[0]) if origin_pos is not None else None
            orig_y = float(origin_pos[1]) if origin_pos is not None else None
            conn.execute(
                """
                INSERT INTO identities (
                    global_id, site_id, canonical_global_id, lifecycle_state,
                    created_at, last_observed_at, slot_id, primary_camera,
                    origin_x, origin_y, max_displacement, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(global_id) DO UPDATE SET
                    canonical_global_id = excluded.canonical_global_id,
                    lifecycle_state = excluded.lifecycle_state,
                    last_observed_at = excluded.last_observed_at,
                    slot_id = excluded.slot_id,
                    primary_camera = excluded.primary_camera,
                    max_displacement = excluded.max_displacement,
                    updated_at = excluded.updated_at;
                """,
                (
                    global_id, self.site_id, canonical_global_id, str(lifecycle_state),
                    float(created_at), float(last_observed_at), slot_id, primary_camera,
                    orig_x, orig_y, float(max_displacement), now,
                ),
            )
            conn.commit()

    def load_active_identities(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT * FROM identities
                WHERE site_id = ? AND lifecycle_state NOT IN ('retired', 'exit_confirmed');
                """,
                (self.site_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def load_parked_identities(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT * FROM identities
                WHERE site_id = ? AND lifecycle_state = 'parked' AND slot_id IS NOT NULL;
                """,
                (self.site_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # --- Identity Aliases ----------------------------------------------------

    def save_alias(self, secondary_id: int, canonical_id: int, reason: str = "") -> None:
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            conn.execute(
                """
                INSERT INTO identity_aliases (secondary_global_id, canonical_global_id, aliased_at, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(secondary_global_id) DO UPDATE SET
                    canonical_global_id = excluded.canonical_global_id,
                    aliased_at = excluded.aliased_at,
                    reason = excluded.reason;
                """,
                (secondary_id, canonical_id, now, reason),
            )
            conn.execute(
                "UPDATE identities SET canonical_global_id = ?, updated_at = ? WHERE global_id = ?;",
                (canonical_id, now, secondary_id),
            )
            conn.commit()

    # --- Identity Events Audit -----------------------------------------------

    def append_event(self, timestamp: float, frame_sequence: int, event_type: str,
                     global_id: int | None, camera_id: str = "", detail: str = "",
                     evidence: dict | None = None) -> int:
        with self._lock:
            conn = self._get_connection()
            evidence_json = json.dumps(evidence or {})
            cursor = conn.execute(
                """
                INSERT INTO identity_events (timestamp, frame_sequence, event_type, global_id, camera_id, detail, evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (float(timestamp), int(frame_sequence), str(event_type), global_id, camera_id, detail, evidence_json),
            )
            conn.commit()
            return cursor.lastrowid

    # --- Kinematic Checkpoints -----------------------------------------------

    def save_checkpoint(self, global_id: int, timestamp: float, pos: tuple[float, float],
                        vel: tuple[float, float], cov: np.ndarray, slot_id: str | None = None,
                        camera_id: str = "", gallery_samples: list[np.ndarray] | None = None) -> None:
        with self._lock:
            conn = self._get_connection()
            cov_arr = np.asarray(cov, dtype=float)
            gallery_json = json.dumps([s.tolist() for s in (gallery_samples or [])[:4]])
            conn.execute(
                """
                INSERT INTO identity_checkpoints (
                    global_id, timestamp, pos_x, pos_y, vel_x, vel_y,
                    cov_00, cov_01, cov_10, cov_11, slot_id, camera_id, gallery_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    global_id, float(timestamp), float(pos[0]), float(pos[1]),
                    float(vel[0]), float(vel[1]),
                    float(cov_arr[0, 0]), float(cov_arr[0, 1]),
                    float(cov_arr[1, 0]), float(cov_arr[1, 1]),
                    slot_id, camera_id, gallery_json,
                ),
            )
            conn.commit()

    def latest_checkpoint(self, global_id: int) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT * FROM identity_checkpoints
                WHERE global_id = ?
                ORDER BY timestamp DESC LIMIT 1;
                """,
                (global_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Sessions & Reservations (Optimistic Concurrency) --------------------

    def save_session(self, session_id: str, global_vehicle_id: int | None = None,
                     state: str = "WAITING_FOR_SCAN", target_spot_id: str | None = None,
                     parked_spot_id: str | None = None, qr_token_hash: str | None = None,
                     qr_expires_at: float | None = None, claimed_at: float | None = None,
                     expected_revision: int | None = None) -> int:
        """Create or update a driver session using optimistic concurrency via revision."""
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            cursor = conn.execute("SELECT revision FROM sessions WHERE session_id = ?;", (session_id,))
            row = cursor.fetchone()
            if row is None:
                new_rev = 1
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, site_id, global_vehicle_id, state,
                        target_spot_id, parked_spot_id, qr_token_hash, qr_expires_at,
                        revision, claimed_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        session_id, self.site_id, global_vehicle_id, state,
                        target_spot_id, parked_spot_id, qr_token_hash, qr_expires_at,
                        new_rev, claimed_at, now,
                    ),
                )
            else:
                curr_rev = int(row["revision"])
                if expected_revision is not None and curr_rev != expected_revision:
                    raise ValueError(f"Optimistic lock failure: expected revision {expected_revision}, got {curr_rev}")
                new_rev = curr_rev + 1
                conn.execute(
                    """
                    UPDATE sessions SET
                        global_vehicle_id = COALESCE(?, global_vehicle_id),
                        state = ?,
                        target_spot_id = ?,
                        parked_spot_id = ?,
                        qr_token_hash = COALESCE(?, qr_token_hash),
                        qr_expires_at = COALESCE(?, qr_expires_at),
                        claimed_at = COALESCE(?, claimed_at),
                        revision = ?,
                        updated_at = ?
                    WHERE session_id = ?;
                    """,
                    (
                        global_vehicle_id, state, target_spot_id, parked_spot_id,
                        qr_token_hash, qr_expires_at, claimed_at, new_rev, now, session_id,
                    ),
                )
            conn.commit()
            return new_rev

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Epoch Recording -----------------------------------------------------

    def record_epoch_start(self, epoch_id: str, runtime_id: str, source_mode: str,
                           config_hash: str = "") -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO runtime_epochs (epoch_id, runtime_id, site_id, source_mode, started_at, config_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(epoch_id) DO NOTHING;
                """,
                (epoch_id, runtime_id, self.site_id, source_mode, time.time(), config_hash),
            )
            conn.commit()
