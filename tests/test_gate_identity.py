"""Phase B invariant tests — gate-controlled minting, GID monotonicity, and soft reset.

Tests the plan's core identity invariants:
- GID is monotonic after reset, restart, and retire (never reuse).
- Mid-lot blob does not mint when require_entry_gate=True.
- Entry gate in wrong direction does not mint.
- Soft reset preserves parked identity and GID counter.
- Crossing detector detects gate crossings with anti-duplicate.
"""

import numpy as np
import pytest

from techgar.config_world import IdentityConfig, AssociationConfig
from techgar.crossing import CrossingDetector, CrossingEvent, GateDefinition
from techgar.registry import GlobalIdentityRegistry
from techgar.states import LifecycleState
from techgar.topology import CameraTopology, CameraZone, TopologyEdge

from conftest import Rig, make_observation, _rect


# ---------------------------------------------------------------------------
# Crossing detector unit tests
# ---------------------------------------------------------------------------

class TestCrossingDetector:
    """Gate crossing detection with direction validation and anti-duplicate."""

    @pytest.fixture
    def entry_gate(self):
        return GateDefinition(
            gate_id="entry_cam1",
            polygon=np.array([[0, 0], [5, 0], [5, 10], [0, 10]], dtype=float),
            inward_direction=np.array([1.0, 0.0]),   # inward = positive x
            gate_type="entry",
            camera_id="cam1",
        )

    @pytest.fixture
    def exit_gate(self):
        return GateDefinition(
            gate_id="exit_cam1",
            polygon=np.array([[40, 0], [45, 0], [45, 10], [40, 10]], dtype=float),
            inward_direction=np.array([1.0, 0.0]),
            gate_type="exit",
            camera_id="cam1",
        )

    def test_entry_crossing_valid_direction(self, entry_gate):
        """Vehicle moving inward through entry gate should be detected."""
        detector = CrossingDetector(gates=[entry_gate])
        event = detector.has_entry_crossing(
            prev_pos=np.array([-1.0, 5.0]),
            curr_pos=np.array([3.0, 5.0]),
            camera_id="cam1",
            timestamp=1.0,
        )
        assert event is not None
        assert event.direction == "entry"
        assert event.gate_id == "entry_cam1"

    def test_entry_crossing_wrong_direction_rejected(self, entry_gate):
        """Vehicle moving outward through entry gate should NOT be detected."""
        detector = CrossingDetector(gates=[entry_gate])
        event = detector.has_entry_crossing(
            prev_pos=np.array([3.0, 5.0]),
            curr_pos=np.array([-1.0, 5.0]),  # moving outward (negative x)
            camera_id="cam1",
            timestamp=1.0,
        )
        assert event is None

    def test_exit_crossing_valid_direction(self, exit_gate):
        """Vehicle moving outward through exit gate should be detected."""
        detector = CrossingDetector(gates=[exit_gate])
        # inward_direction is [1,0] (positive x = inward).
        # For exit gate, valid_direction checks dot < 0, meaning vehicle must
        # move in *negative* x direction (outward from facility).
        event = detector.has_exit_crossing(
            prev_pos=np.array([43.0, 5.0]),    # inside exit gate
            curr_pos=np.array([39.0, 5.0]),    # moving outward (negative x)
            camera_id="cam1",
            timestamp=1.0,
        )
        assert event is not None
        assert event.direction == "exit"

    def test_anti_duplicate_same_crossing(self, entry_gate):
        """Same crossing should not produce two events."""
        detector = CrossingDetector(gates=[entry_gate])
        event1 = detector.has_entry_crossing(
            prev_pos=np.array([-1.0, 5.0]),
            curr_pos=np.array([3.0, 5.0]),
            camera_id="cam1",
            timestamp=1.0,
            track_id=1,
        )
        event2 = detector.has_entry_crossing(
            prev_pos=np.array([-1.0, 5.0]),
            curr_pos=np.array([3.0, 5.0]),
            camera_id="cam1",
            timestamp=1.0,
            track_id=1,
        )
        assert event1 is not None
        assert event2 is None  # duplicate blocked

    def test_anti_duplicate_key_expires(self, entry_gate):
        """After TTL, same crossing event key should be allowed again."""
        detector = CrossingDetector(gates=[entry_gate], key_ttl=5.0)
        event1 = detector.has_entry_crossing(
            prev_pos=np.array([-1.0, 5.0]),
            curr_pos=np.array([3.0, 5.0]),
            camera_id="cam1",
            timestamp=1.0,
            track_id=1,
        )
        assert event1 is not None
        # Same crossing 10 seconds later — key has expired
        event2 = detector.has_entry_crossing(
            prev_pos=np.array([-1.0, 5.0]),
            curr_pos=np.array([3.0, 5.0]),
            camera_id="cam1",
            timestamp=11.0,
            track_id=1,
        )
        assert event2 is not None

    def test_no_crossing_when_outside_gate(self, entry_gate):
        """Movement entirely outside gate polygon should not trigger crossing."""
        detector = CrossingDetector(gates=[entry_gate])
        event = detector.has_entry_crossing(
            prev_pos=np.array([20.0, 5.0]),
            curr_pos=np.array([25.0, 5.0]),
            camera_id="cam1",
            timestamp=1.0,
        )
        assert event is None

    def test_wrong_camera_ignored(self, entry_gate):
        """Gate for cam1 should not fire for cam2 observations."""
        detector = CrossingDetector(gates=[entry_gate])
        event = detector.has_entry_crossing(
            prev_pos=np.array([-1.0, 5.0]),
            curr_pos=np.array([3.0, 5.0]),
            camera_id="cam2",
            timestamp=1.0,
        )
        assert event is None


# ---------------------------------------------------------------------------
# GID monotonicity invariants
# ---------------------------------------------------------------------------

class TestGIDMonotonicity:
    """GID must increase monotonically after reset, restart, and retire."""

    def test_gid_monotonic_after_hard_reset(self, topology):
        """After hard reset, GID counter does NOT reset to 0."""
        rig = Rig(topology)
        # Drive first vehicle to get GID
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        first_gid = rig.single_live_gid()
        assert first_gid >= 1

        # Hard reset
        rig.registry.reset(10.0, 100)

        # Drive second vehicle — should get higher GID
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1", start=11.0)
        second_gid = rig.single_live_gid()
        assert second_gid > first_gid, (
            f"GID after reset ({second_gid}) must be > GID before reset ({first_gid})")

    def test_gid_monotonic_after_retire(self, topology):
        """A retired GID is never reused."""
        rig = Rig(topology)
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        first_gid = rig.single_live_gid()

        # Force retire
        state = rig.registry.get(first_gid)
        state.lifecycle_state = LifecycleState.EXIT_CONFIRMED
        rig.step(timestamp=20.0)  # sweep will retire it

        # New vehicle
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1", start=21.0)
        second_gid = rig.single_live_gid()
        assert second_gid > first_gid

    def test_gid_monotonic_after_soft_reset(self, topology):
        """Soft reset preserves GID counter."""
        rig = Rig(topology)
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        first_gid = rig.single_live_gid()

        # Soft reset
        result = rig.registry.soft_reset(10.0, 100)
        assert result["soft_reset"] is True

        # New vehicle should get higher GID
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1", start=11.0)
        live = rig.registry.live()
        new_gids = [s.global_id for s in live if s.lifecycle_state is not LifecycleState.RECOVERY_PENDING]
        assert any(g > first_gid for g in new_gids), (
            f"GID after soft_reset must be > {first_gid}, got {new_gids}")


# ---------------------------------------------------------------------------
# Gate-controlled minting
# ---------------------------------------------------------------------------

class TestGateControlledMinting:
    """When require_entry_gate=True, only entry-gate observations mint GIDs."""

    def _make_gated_topology(self):
        """Topology with entry corridor at x∈[0,5]."""
        cam1 = CameraZone(
            camera_id="cam1",
            fov_polygon=_rect(0, 0, 100, 60),
            exit_polygons={},
            entry_polygons={"external": _rect(0, 10, 5, 20)},
        )
        return CameraTopology(zones={"cam1": cam1})

    def test_midlot_blob_does_not_mint(self):
        """Detection in the middle of the lot must NOT mint when gate required."""
        topology = self._make_gated_topology()
        config = IdentityConfig(require_entry_gate=True, t_maturity=0.0, n_maturity=1)
        rig = Rig(topology, identity_config=config)
        # Observation at (50, 30) — middle of lot, not near entry corridor
        rig.drive(50.0, 30.0, 0.1, "cam1")
        rig.drive(52.0, 30.0, 0.2, "cam1")
        rig.drive(54.0, 30.0, 0.3, "cam1")

        live = rig.registry.live()
        assert len(live) == 0, (
            f"Mid-lot blob should NOT mint when require_entry_gate=True, "
            f"but got {len(live)} identities")

    def test_entry_corridor_observation_mints(self):
        """Detection near entry corridor SHOULD mint when gate required."""
        topology = self._make_gated_topology()
        config = IdentityConfig(require_entry_gate=True, t_maturity=0.0, n_maturity=1)
        rig = Rig(topology, identity_config=config)
        # Observation at (3, 15) — inside entry corridor [0,5] x [10,20]
        rig.drive(3.0, 15.0, 0.1, "cam1")
        rig.drive(5.0, 15.0, 0.2, "cam1")
        rig.drive(7.0, 15.0, 0.3, "cam1")

        live = rig.registry.live()
        assert len(live) >= 1, (
            f"Entry corridor observation should mint when require_entry_gate=True, "
            f"but got {len(live)} identities")

    def test_default_config_mints_anywhere(self, topology):
        """Default config (require_entry_gate=False) mints at any position."""
        rig = Rig(topology)
        rig.drive_n([(50, 30), (52, 30), (54, 30)], "cam1")
        live = rig.registry.live()
        assert len(live) >= 1, "Default config should mint at any position"


# ---------------------------------------------------------------------------
# Soft reset preserves parked identity
# ---------------------------------------------------------------------------

class TestSoftReset:
    """Soft reset must preserve parked identities with slot ownership."""

    def test_soft_reset_keeps_parked_identity(self, topology):
        """Parked identity survives soft reset."""
        rig = Rig(topology)
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        gid = rig.single_live_gid()
        # Manually park
        state = rig.registry.get(gid)
        state.lifecycle_state = LifecycleState.PARKED
        state.slot_id = "A01"

        result = rig.registry.soft_reset(10.0, 100)
        assert result["parked_kept"] == 1
        # Verify parked identity still exists
        state = rig.registry.get(gid)
        assert state is not None
        assert state.lifecycle_state is LifecycleState.PARKED
        assert state.slot_id == "A01"

    def test_soft_reset_moves_active_to_recovery_pending(self, topology):
        """Active moving identity becomes RECOVERY_PENDING after soft reset."""
        rig = Rig(topology)
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        gid = rig.single_live_gid()

        result = rig.registry.soft_reset(10.0, 100)
        assert result["recovery_pending"] >= 1
        state = rig.registry.get(gid)
        assert state is not None
        assert state.lifecycle_state is LifecycleState.RECOVERY_PENDING

    def test_soft_reset_retires_provisional(self, topology):
        """PROVISIONAL identity is immediately retired on soft reset."""
        config = IdentityConfig(t_maturity=100.0, n_maturity=100)  # never matures
        rig = Rig(topology, identity_config=config)
        rig.drive(10.0, 30.0, 0.1, "cam1")
        rig.drive(12.0, 30.0, 0.2, "cam1")

        result = rig.registry.soft_reset(10.0, 100)
        assert result["retired"] >= 1
        assert len(rig.registry.live()) == 0

    def test_close_all_does_not_reset_gid_sequence(self, topology):
        """close-all (hard reset) does not reset the GID sequence counter."""
        rig = Rig(topology)
        rig.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        gid_before = rig.single_live_gid()
        counter_before = rig.registry._next_global_id

        rig.registry.reset(10.0, 100)
        counter_after = rig.registry._next_global_id
        assert counter_after == counter_before, (
            f"Hard reset must NOT reset GID counter: was {counter_before}, now {counter_after}")
