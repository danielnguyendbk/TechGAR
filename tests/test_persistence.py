"""Phase D persistence tests — SQLite WAL, schema migrations, monotonic GID sequence,
kinematic checkpoints, session optimistic concurrency, and crash recovery.
"""

import time
import numpy as np
import pytest

from techgar.config_world import IdentityConfig
from techgar.persistence import PersistenceStore
from techgar.registry import GlobalIdentityRegistry
from techgar.states import LifecycleState
from conftest import Rig, _rect


class TestPersistenceStoreBasics:
    """Test table creation, migrations, and monotonic GID sequence."""

    def test_migrations_create_all_tables(self, tmp_path):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")
        conn = store._get_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {
            "schema_migrations",
            "identity_sequence",
            "identities",
            "identity_aliases",
            "identity_events",
            "identity_checkpoints",
            "sessions",
            "reservations",
            "runtime_epochs",
        }
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"
        store.close()

    def test_monotonic_gid_sequence_never_reuses(self, tmp_path):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")

        # Allocate 3 IDs
        id1 = store.next_global_id()
        id2 = store.next_global_id()
        id3 = store.next_global_id()
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
        store.close()

        # Reopen database (simulate process restart)
        store2 = PersistenceStore(db_file, site_id="site_alpha")
        id4 = store2.next_global_id()
        assert id4 == 4, f"After restart, GID must continue monotonically: got {id4}, expected 4"
        store2.close()

    def test_independent_site_sequences(self, tmp_path):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file)
        site_a_1 = store.next_global_id(site_id="site_A")
        site_b_1 = store.next_global_id(site_id="site_B")
        site_a_2 = store.next_global_id(site_id="site_A")
        assert site_a_1 == 1
        assert site_b_1 == 1
        assert site_a_2 == 2
        store.close()


class TestKinematicCheckpointsAndIdentities:
    """Test identity save/load and kinematic checkpointing."""

    def test_save_and_load_parked_identities(self, tmp_path):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")

        store.save_identity(
            global_id=1,
            lifecycle_state="parked",
            created_at=100.0,
            last_observed_at=105.0,
            slot_id="A01",
            primary_camera="cam1",
            origin_pos=(10.0, 20.0),
        )
        store.save_identity(
            global_id=2,
            lifecycle_state="active",
            created_at=100.0,
            last_observed_at=106.0,
            primary_camera="cam1",
            origin_pos=(15.0, 25.0),
        )

        parked = store.load_parked_identities()
        assert len(parked) == 1
        assert parked[0]["global_id"] == 1
        assert parked[0]["slot_id"] == "A01"

        active = store.load_active_identities()
        assert len(active) == 2  # parked and active are both non-retired
        store.close()

    def test_save_and_restore_checkpoint(self, tmp_path):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")

        cov = np.array([[0.05, 0.01], [0.01, 0.05]])
        store.save_checkpoint(
            global_id=1,
            timestamp=102.5,
            pos=(12.0, 22.0),
            vel=(1.5, 0.2),
            cov=cov,
            slot_id=None,
            camera_id="cam1",
        )

        chk = store.latest_checkpoint(global_id=1)
        assert chk is not None
        assert chk["global_id"] == 1
        assert abs(chk["pos_x"] - 12.0) < 1e-4
        assert abs(chk["pos_y"] - 22.0) < 1e-4
        assert abs(chk["cov_00"] - 0.05) < 1e-4
        store.close()


class TestSessionOptimisticConcurrency:
    """Test session lifecycle and revision-based optimistic concurrency."""

    def test_session_revision_increments_monotonically(self, tmp_path):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")

        rev1 = store.save_session("sess_01", global_vehicle_id=1, state="WAITING_FOR_SCAN")
        assert rev1 == 1

        rev2 = store.save_session("sess_01", target_spot_id="A01", state="NAVIGATING",
                                  expected_revision=1)
        assert rev2 == 2

        # Conflicting update with stale revision should raise ValueError
        with pytest.raises(ValueError, match="Optimistic lock failure"):
            store.save_session("sess_01", target_spot_id="A02", expected_revision=1)

        sess = store.get_session("sess_01")
        assert sess is not None
        assert sess["revision"] == 2
        assert sess["target_spot_id"] == "A01"
        assert sess["state"] == "NAVIGATING"
        store.close()


class TestRegistryPersistenceIntegration:
    """Test GlobalIdentityRegistry backed by PersistenceStore."""

    def test_registry_mint_uses_persistent_sequence(self, tmp_path, topology):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")

        # Initialize registry with store
        registry = GlobalIdentityRegistry(topology=topology, store=store, site_id="site_alpha")
        rig = Rig(topology, registry=registry)

        # Drive vehicle at 8 m/s (< v_max_world = 12 m/s)
        rig.drive_n([(10, 30), (10.8, 30), (11.6, 30)], "cam1")
        gid1 = rig.single_live_gid()
        assert gid1 == 1

        # Check that store has sequence 1
        assert store.current_global_id("site_alpha") == 1

        # Simulate restart: new registry instance with same DB
        registry2 = GlobalIdentityRegistry(topology=topology, store=store, site_id="site_alpha")
        rig2 = Rig(topology, registry=registry2)

        # Drive new vehicle — GID must continue to 2!
        rig2.drive_n([(10, 30), (10.8, 30), (11.6, 30)], "cam1", start=10.0)
        gid2 = rig2.single_live_gid()
        assert gid2 == 2, f"Expected monotonic GID 2 after restart, got {gid2}"
        store.close()

    def test_startup_recovery_restores_parked_identities(self, tmp_path, topology):
        db_file = tmp_path / "test_techgar.db"
        store = PersistenceStore(db_file, site_id="site_alpha")

        # Registry 1: park a vehicle
        registry1 = GlobalIdentityRegistry(topology=topology, store=store, site_id="site_alpha")
        rig1 = Rig(topology, registry=registry1)
        rig1.drive_n([(10, 30), (12, 30), (14, 30)], "cam1")
        gid = rig1.single_live_gid()
        registry1.mark_parked(gid, "B02", 5.0, 50)

        # Simulate restart with fresh registry
        registry2 = GlobalIdentityRegistry(topology=topology, store=store, site_id="site_alpha")
        state = registry2.get(gid)
        assert state is not None, "Parked vehicle must be restored on startup"
        assert state.lifecycle_state == LifecycleState.PARKED
        assert state.slot_id == "B02"
        store.close()
