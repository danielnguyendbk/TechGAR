"""Runtime, session and operator HTTP boundaries."""

from fastapi.testclient import TestClient

from techgar.api import RuntimeService, create_app
from test_snapshot_contract import make_snapshot


def client_with_snapshot():
    service = RuntimeService()
    service.set_snapshot(make_snapshot())
    return TestClient(create_app(service)), service


def test_runtime_snapshot_endpoint_returns_versioned_contract():
    client, _ = client_with_snapshot()
    response = client.get("/api/runtime/snapshot")
    assert response.status_code == 200
    assert response.json()["schema_version"] in ("1.0", "2.0")
    assert response.json()["frame_index"] == 7


def test_session_claim_is_idempotent_and_navigation_is_explicit():
    client, _ = client_with_snapshot()
    first = client.post("/api/sessions/S42/claim", json={"global_vehicle_id": 17})
    second = client.post("/api/sessions/S42/claim", json={"global_vehicle_id": 999})
    assert first.status_code == second.status_code == 200
    assert first.json()["globalVehicleId"] == second.json()["globalVehicleId"] == 17
    session = client.get("/api/sessions/S42").json()
    assert session["targetSpotId"] is None
    selected = client.post("/api/sessions/S42/select-spot", json={"spot_id": "B04"})
    assert selected.status_code == 200
    assert selected.json()["state"] == "NAVIGATING"


def test_gate_editor_requires_exactly_six_world_points():
    client, service = client_with_snapshot()
    bad = client.post("/api/runtime/gates", json={"points": [[0, 0]]})
    assert bad.status_code == 422
    points = [[float(index), float(index + 1)] for index in range(6)]
    good = client.post("/api/runtime/gates", json={"points": points})
    assert good.status_code == 200
    assert service.gate_points == [tuple(point) for point in points]


def test_reset_endpoint_is_single_explicit_action():
    client, _ = client_with_snapshot()
    result = client.post("/api/runtime/reset-identities", json={"include_sessions": False})
    assert result.status_code == 200
    assert result.json()["retired_identities"] == 1
    assert client.get("/api/runtime/snapshot").json()["vehicles"] == []

