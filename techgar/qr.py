"""Phase E1 — Dynamic kiosk QR tokens with 60-second lifecycle.

Implements the plan's kiosk invariant:
- Dynamic QR generated upon vehicle entry activation.
- Token is cryptographically random, hash-only in SQLite, with strict 60s TTL.
- Driver scan triggers idempotent claim: subsequent scans within TTL return the same session.
- Claiming binds the session to the activated vehicle's Global ID without client providing GID.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .persistence import PersistenceStore


def hash_token(raw_token: str) -> str:
    """SHA-256 hash of the raw token string."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


@dataclass
class QRTokenRecord:
    token: str
    token_hash: str
    global_vehicle_id: int
    expires_at: float
    session_id: str | None = None
    claimed_at: float | None = None

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_claimed(self) -> bool:
        return self.session_id is not None and self.claimed_at is not None


class QRTokenManager:
    """Thread-safe generator and claim validator for kiosk QR tokens."""

    def __init__(self, store: PersistenceStore | None = None, site_id: str = "default_site",
                 default_ttl: float = 60.0) -> None:
        self.store = store
        self.site_id = site_id
        self.default_ttl = float(default_ttl)
        self._tokens: dict[str, QRTokenRecord] = {}  # keyed by token_hash
        self._vehicle_tokens: dict[int, str] = {}    # global_id -> token_hash
        self._lock = threading.RLock()

    def generate_token(self, global_vehicle_id: int, ttl: float | None = None) -> str:
        """Generate a fresh URL-safe token for a vehicle activated at entry gate."""
        with self._lock:
            now = time.time()
            lifespan = float(ttl if ttl is not None else self.default_ttl)
            raw_token = secrets.token_urlsafe(16)
            token_h = hash_token(raw_token)
            record = QRTokenRecord(
                token=raw_token,
                token_hash=token_h,
                global_vehicle_id=global_vehicle_id,
                expires_at=now + lifespan,
            )
            self._tokens[token_h] = record
            self._vehicle_tokens[global_vehicle_id] = token_h
            return raw_token

    def get_token_for_vehicle(self, global_vehicle_id: int) -> str | None:
        """Get the active unexpired raw token for a vehicle, if available."""
        with self._lock:
            token_h = self._vehicle_tokens.get(global_vehicle_id)
            if token_h is None:
                return None
            record = self._tokens.get(token_h)
            if record is None or record.is_expired:
                return None
            return record.token

    def claim_token(self, raw_token: str, now: float | None = None) -> tuple[str, int]:
        """Claim a QR token idempotently.

        Returns (session_id, global_vehicle_id).
        Raises KeyError if token is unknown.
        Raises ValueError if token is expired.
        """
        moment = time.time() if now is None else float(now)
        token_h = hash_token(raw_token)
        with self._lock:
            record = self._tokens.get(token_h)
            if record is None:
                raise KeyError(f"Invalid QR token: {raw_token[:6]}...")
            if moment > record.expires_at:
                raise ValueError("QR token has expired")

            # Idempotent: return existing session if already claimed
            if record.is_claimed:
                return str(record.session_id), record.global_vehicle_id

            session_id = f"sess_{secrets.token_hex(8)}"
            record.session_id = session_id
            record.claimed_at = moment

            if self.store is not None:
                self.store.save_session(
                    session_id=session_id,
                    global_vehicle_id=record.global_vehicle_id,
                    state="SELECTING_SPOT",
                    qr_token_hash=token_h,
                    qr_expires_at=record.expires_at,
                    claimed_at=moment,
                )

            return session_id, record.global_vehicle_id

    def cleanup_expired(self, now: float | None = None) -> int:
        """Remove expired tokens to prevent memory growth."""
        moment = time.time() if now is None else float(now)
        with self._lock:
            expired_hashes = [h for h, rec in self._tokens.items() if moment > rec.expires_at]
            for h in expired_hashes:
                rec = self._tokens.pop(h, None)
                if rec is not None and self._vehicle_tokens.get(rec.global_vehicle_id) == h:
                    self._vehicle_tokens.pop(rec.global_vehicle_id, None)
            return len(expired_hashes)
