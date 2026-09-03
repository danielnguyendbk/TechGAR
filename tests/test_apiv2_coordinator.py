"""Tests for API v2, RuntimeCoordinator, QR token lifecycle, and atomic slot reservations."""

import time
import pytest
from fastapi.testclient import TestClient

from techgar.api import create_app
from techgar.coordinator import RuntimeCoordinator
from techgar.persistence import PersistenceStore
from techgar.qr import QRTokenManager
from techgar.reservation import ReservationManager, SlotConflictError
from test_snapshot_contract import make_snapshot


class TestCoordinatorStateMachine:
    """State machine v2 invariant tests."""

    def test_valid_forward_flow(self):
        coord = RuntimeCoordinator()
        token = coord.qr_manager.generate_token(global_vehicle_id=10)
        session = coord.claim_via_token(token)
        assert session.state == "SELECTING_SPOT"

        coord.select_spot_reservation(session.session_id, "A01")
        assert session.state == "NAVIGATING"
        assert session.target_spot_id == "A01"

        # Internal slot engine confirms park
        coord.on_slot_parked(global_id=10, physical_slot_id="A01")
        assert session.state == "PARKED"
        assert session.parked_spot_id == "A01"

        coord.request_exit(session.session_id)
        assert session.state == "EXIT_NAVIGATION"

        coord.on_physical_exit(global_id=10)
        assert session.state == "CLOSED"

    def test_forbid_backwards_transition_exit_to_parked(self):
        coord = RuntimeCoordinator()
        token = coord.qr_manager.generate_token(global_vehicle_id=10)
        session = coord.claim_via_token(token)
        coord.select_spot_reservation(session.session_id, "A01")
        coord.on_slot_parked(global_id=10, physical_slot_id="A01")
        coord.request_exit(session.session_id)
        assert session.state == "EXIT_NAVIGATION"

        # Backwards transition from EXIT_NAVIGATION to PARKED is strictly forbidden
        with pytest.raises(ValueError, match="Invalid state transition"):
            coord._transition_session(session, "PARKED")


class TestQRTokenLifecycle:
    """QR tokens with 60s TTL and idempotent claim."""

    def test_qr_idempotent_claim_within_ttl(self):
        mgr = QRTokenManager()
        token = mgr.generate_token(global_vehicle_id=42, ttl=60.0)

        # First claim
        sess_id1, gid1 = mgr.claim_token(token)
        assert gid1 == 42

        # Second claim with same token must return the SAME session (idempotent)
        sess_id2, gid2 = mgr.claim_token(token)
        assert sess_id1 == sess_id2
        assert gid1 == gid2

    def test_qr_expired_token_rejected(self):
        mgr = QRTokenManager()
        token = mgr.generate_token(global_vehicle_id=42, ttl=0.1)
        time.sleep(0.15)
        with pytest.raises(ValueError, match="expired"):
            mgr.claim_token(token)


class TestReservationManager:
    """Atomic slot reservations and slot swap."""

    def test_slot_collision_rejected(self):
        mgr = ReservationManager(lease_duration=300.0)
        mgr.acquire_lease("session_1", "A01", global_vehicle_id=1)

        # Another session attempting to lease the same slot must fail
        with pytest.raises(SlotConflictError):
            mgr.acquire_lease("session_2", "A01", global_vehicle_id=2)

    def test_auto_swap_on_parked_different_slot(self):
        mgr = ReservationManager(lease_duration=300.0)
        mgr.acquire_lease("session_1", "A01", global_vehicle_id=1)
        assert mgr.get_lease_for_session("session_1").slot_id == "A01"

        # Vehicle physically parked in B02 instead
        new_lease = mgr.auto_swap_on_parked("session_1", "B02")
        assert new_lease.slot_id == "B02"
        assert mgr.get_lease_for_slot("A01") is None
        assert mgr.get_lease_for_slot("B02").session_id == "session_1"


class TestAPIv2Endpoints:
    """FastAPI v2 contract and endpoint tests."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_health_and_ready_endpoints(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["schema_version"] == "2.0"

        resp_ready = client.get("/api/ready")
        assert resp_ready.status_code == 200
        assert resp_ready.json()["ready"] is True

    def test_soft_reset_endpoint(self, client):
        resp = client.post("/api/runtime/soft-reset")
        assert resp.status_code == 200
        assert resp.json()["soft_reset"] is True

    def test_close_all_requires_confirmation(self, client):
        # Without confirm -> 400 Bad Request
        resp_bad = client.post("/api/runtime/close-all", json={"confirm": False})
        assert resp_bad.status_code == 400

        resp_good = client.post("/api/runtime/close-all", json={"confirm": True})
        assert resp_good.status_code == 200

    def test_reservation_put_and_delete(self, client):
        # Create a session
        client.post("/api/sessions/S10/claim", json={"global_vehicle_id": 5})

        # Reserve a spot
        res = client.put("/api/sessions/S10/reservation", json={"spot_id": "C01"})
        assert res.status_code == 200
        assert res.json()["targetSpotId"] == "C01"
        assert res.json()["state"] == "NAVIGATING"

        # Cancel reservation
        del_res = client.delete("/api/sessions/S10/reservation")
        assert del_res.status_code == 200
        assert del_res.json()["targetSpotId"] is None
