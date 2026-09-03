"""Backend/frontend JSON contract and reset behavior."""

from techgar.api import RuntimeService
from techgar.snapshot import RuntimeSnapshot, SlotView, VehicleView
from techgar.states import DisplayState, LifecycleState
from techgar.units import WORLD_FRAME_NAME


def make_snapshot(sequence: int = 7) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        sequence=sequence,
        timestamp=100.0,
        world_frame=WORLD_FRAME_NAME,
        vehicles=(VehicleView(
            global_id=17,
            display_state=DisplayState.PARKED,
            lifecycle_state=LifecycleState.PARKED,
            world_position=(50.0, 120.0),
            uncertainty=0.2,
            velocity=(0.0, 0.0),
            camera_id="C1",
            slot_id="B04",
            session_ids=("S42",),
            last_observed=90.0,
            footprint=((49.0, 119.0), (51.0, 119.0), (51.0, 121.0), (49.0, 121.0)),
            observed=False,
            stale_seconds=10.0,
            display_hold_seconds=2.5,
        ),),
        slots=(SlotView("B04", "occupied", 17, 0.91, 4.2, 0.95),),
        camera_health={"C1": {"frames": 20, "last_timestamp": 99.5}},
        identity_events=(),
        published_at=100.1,
        slot_layout=({"slot_id": "B04", "camera_id": "shared",
                      "polygon": [[49, 119], [51, 119], [51, 121], [49, 121]]},),
    )


def test_snapshot_exposes_plan_4_contract_and_legacy_aliases():
    payload = make_snapshot().to_dict()
    assert payload["schema_version"] == "1.0"
    assert payload["frame_index"] == 7
    assert payload["published_at"] == 100.1
    assert payload["vehicles"][0]["parked_slot_id"] == "B04"
    assert payload["vehicles"][0]["observed"] is False
    assert payload["parking_slots"][0]["occupied"] is True
    assert payload["slot_layout"][0]["slot_id"] == "B04"
    assert payload["cameras"]["C1"]["online"] is True
    assert payload["sequence"] == payload["frame_index"]
    assert payload["slots"][0]["slot_id"] == "B04"


def test_runtime_service_rejects_snapshot_regression():
    service = RuntimeService()
    service.set_snapshot(make_snapshot(8))
    try:
        service.set_snapshot(make_snapshot(7))
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("frame_index regression must be rejected")


def test_runtime_reset_clears_markers_and_restarts_contract_sequence():
    service = RuntimeService()
    service.set_snapshot(make_snapshot())
    result = service.reset(include_sessions=False)
    assert result["retired_identities"] == 1
    payload = service.current_snapshot().to_dict()
    assert payload["vehicles"] == []
    assert payload["frame_index"] == 8

