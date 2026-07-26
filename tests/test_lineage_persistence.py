"""Lineage state must survive a proxy restart.

Without persistence every restart drops prev_orig/prev_fwd for all live
sessions, so overlay_cached_prefix has nothing to replay and each session
rewrites its whole history at the provider's write rate. These tests pin the
round trip and the guards that keep a restored snapshot from being worse than a
cold start.
"""

from __future__ import annotations

import gzip
import json
import os
import time

import pytest

from headroom.cache.lineage_persistence import (
    DIR_ENV_VAR,
    SCHEMA_VERSION,
    LineagePersistence,
    default_state_path,
    persistence_enabled,
)
from headroom.cache.prefix_tracker import PrefixFreezeConfig, SessionTrackerStore


def _messages(*texts: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": t}]} for t in texts]


def _store() -> SessionTrackerStore:
    return SessionTrackerStore(default_config=PrefixFreezeConfig(enabled=True))


def _served_turn(store, session_id, provider, originals, forwarded):
    """Resolve a tracker and record a response, as one served turn would."""
    tracker = store.resolve_tracker(session_id, provider, originals)
    tracker.update_from_response(
        cache_read_tokens=1000,
        cache_write_tokens=100,
        messages=forwarded,
        original_messages=originals,
    )
    return tracker


class TestTrackerRoundTrip:
    def test_message_blobs_survive(self):
        store = _store()
        originals = _messages("first", "second")
        forwarded = _messages("first", "compressed second")
        _served_turn(store, "sess-a", "anthropic", originals, forwarded)

        revived = _store()
        assert revived.restore_state(store.export_state()) == 1

        tracker = revived.get_or_create("sess-a", "anthropic")
        assert tracker.get_last_original_messages() == originals
        assert tracker.get_last_forwarded_messages() == forwarded

    def test_scalars_survive(self):
        store = _store()
        tracker = _served_turn(
            store, "sess-a", "anthropic", _messages("a"), _messages("a")
        )
        tracker._compress_latched = True
        tracker._kept_ewma = 0.42
        tracker._last_head_fingerprint = "head-abc"
        tracker._placed_anchor_depths = [64, 128]

        revived = _store()
        revived.restore_state(store.export_state())
        got = revived.get_or_create("sess-a", "anthropic")

        assert got._turn_number == tracker._turn_number
        assert got._cached_message_count == tracker._cached_message_count
        assert got._cached_token_count == tracker._cached_token_count
        assert got._compress_latched is True
        assert got._kept_ewma == 0.42
        assert got._last_head_fingerprint == "head-abc"
        assert got._placed_anchor_depths == [64, 128]

    def test_hybrid_controller_state_survives(self):
        store = _store()
        tracker = _served_turn(
            store, "sess-a", "anthropic", _messages("a"), _messages("a")
        )
        tracker.hybrid_controller.generation = 3
        tracker.hybrid_controller._warm_turns = 5
        tracker.hybrid_controller._cooldown_remaining = 2

        revived = _store()
        revived.restore_state(store.export_state())
        controller = revived.get_or_create("sess-a", "anthropic").hybrid_controller

        assert controller.generation == 3
        assert controller._warm_turns == 5
        assert controller._cooldown_remaining == 2

    def test_restored_tracker_keeps_its_frozen_prefix(self):
        """The point of the exercise: turn 0 would re-freeze from scratch."""
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("a", "b"), _messages("a", "b"))
        expected = store.get_or_create("sess-a", "anthropic").get_frozen_message_count()
        assert expected > 0

        revived = _store()
        revived.restore_state(store.export_state())
        assert revived.get_or_create("sess-a", "anthropic").get_frozen_message_count() == expected


class TestLineageReconstruction:
    def test_next_turn_matches_the_restored_lineage(self):
        store = _store()
        originals = _messages("first", "second")
        _served_turn(store, "sess-a", "anthropic", originals, originals)

        revived = _store()
        revived.restore_state(store.export_state())

        # The client appends, as client histories always do.
        extended = originals + _messages("third")
        tracker = revived.resolve_tracker("sess-a", "anthropic", extended)

        # Matched the restored lineage rather than starting a fresh one.
        assert tracker.get_last_original_messages() == originals
        assert len(revived._trackers) == 1

    def test_divergent_history_still_starts_a_fresh_lineage(self):
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("first"), _messages("first"))

        revived = _store()
        revived.restore_state(store.export_state())
        revived.resolve_tracker("sess-a", "anthropic", _messages("totally other"))

        assert len(revived._trackers) == 2

    def test_synthetic_keys_do_not_collide_after_restore(self):
        """A reused lineage number would silently merge two conversations."""
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("alpha"), _messages("alpha"))
        _served_turn(store, "sess-a", "anthropic", _messages("beta"), _messages("beta"))
        assert any("\x00" in k for k in store._trackers)

        revived = _store()
        revived.restore_state(store.export_state())
        before = set(revived._trackers)
        revived.resolve_tracker("sess-a", "anthropic", _messages("gamma"))

        new_keys = set(revived._trackers) - before
        assert len(new_keys) == 1
        assert new_keys.isdisjoint(before)

    def test_overflow_tracker_is_not_matched_as_a_lineage(self):
        store = _store()
        tracker = store.get_or_create("sess-a\x00overflow", "anthropic")
        tracker.update_from_response(
            cache_read_tokens=1000,
            cache_write_tokens=100,
            messages=_messages("shared"),
            original_messages=_messages("shared"),
        )

        revived = _store()
        revived.restore_state(store.export_state())
        assert "sess-a\x00overflow" in revived._trackers
        assert revived._lineages.get("sess-a", {}) == {}


class TestExpiry:
    def test_expired_trackers_are_not_exported(self):
        store = _store()
        tracker = _served_turn(
            store, "sess-a", "anthropic", _messages("a"), _messages("a")
        )
        tracker._last_activity = time.time() - 10_000

        assert store.export_state()["trackers"] == {}

    def test_stale_snapshot_entries_are_dropped_on_load(self):
        """A dead lineage must not win a match in resolve_tracker."""
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("a"), _messages("a"))
        state = store.export_state()
        state["trackers"]["sess-a"]["last_activity"] = time.time() - 10_000

        revived = _store()
        assert revived.restore_state(state) == 0
        assert revived._lineages.get("sess-a", {}) == {}

    def test_live_trackers_win_over_the_snapshot(self):
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("snapshot"), _messages("snapshot"))

        revived = _store()
        _served_turn(revived, "sess-a", "anthropic", _messages("live"), _messages("live"))
        revived.restore_state(store.export_state())

        assert revived.get_or_create("sess-a", "anthropic").get_last_original_messages() == _messages("live")


class TestRestoreIsDefensive:
    @pytest.mark.parametrize(
        "blob",
        [
            {},
            {"trackers": None},
            {"trackers": "nonsense"},
            {"trackers": {"sess-a": "not-a-dict"}},
            {"trackers": {42: {"provider": "anthropic"}}},
        ],
    )
    def test_malformed_snapshots_restore_nothing(self, blob):
        revived = _store()
        assert revived.restore_state(blob) == 0

    def test_missing_keys_fall_back_to_a_cold_tracker(self):
        revived = _store()
        assert revived.restore_state({"trackers": {"sess-a": {}}}) == 1
        tracker = revived.get_or_create("sess-a", "anthropic")
        assert tracker.get_last_original_messages() == []
        assert tracker._turn_number == 0

    def test_unknown_hybrid_phase_falls_back_to_cold(self):
        from headroom.proxy.hybrid_mode import HybridPhase

        revived = _store()
        revived.restore_state(
            {"trackers": {"sess-a": {"provider": "anthropic", "hybrid": {"phase": "bogus"}}}}
        )
        controller = revived.get_or_create("sess-a", "anthropic").hybrid_controller
        assert controller.phase == HybridPhase.COLD_PREFIX


class TestFileLayer:
    def test_save_load_round_trip(self, tmp_path):
        persistence = LineagePersistence(tmp_path / "state.json.gz")
        assert persistence.save({"trackers": {"sess-a": {"provider": "anthropic"}}})
        assert persistence.load() == {"trackers": {"sess-a": {"provider": "anthropic"}}}

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert LineagePersistence(tmp_path / "absent.json.gz").load() is None

    def test_corrupt_file_is_discarded(self, tmp_path):
        path = tmp_path / "state.json.gz"
        path.write_bytes(b"not gzip at all")
        assert LineagePersistence(path).load() is None

    def test_truncated_gzip_is_discarded(self, tmp_path):
        path = tmp_path / "state.json.gz"
        persistence = LineagePersistence(path)
        persistence.save({"trackers": {"sess-a": {"provider": "anthropic"}}})
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
        assert persistence.load() is None

    def test_foreign_schema_version_is_ignored(self, tmp_path):
        path = tmp_path / "state.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"version": SCHEMA_VERSION + 99, "state": {"trackers": {}}}, handle)
        assert LineagePersistence(path).load() is None

    def test_snapshot_is_owner_only(self, tmp_path):
        path = tmp_path / "state.json.gz"
        LineagePersistence(path).save({"trackers": {}})
        assert path.stat().st_mode & 0o077 == 0

    def test_failed_write_leaves_the_previous_snapshot_intact(self, tmp_path):
        path = tmp_path / "state.json.gz"
        persistence = LineagePersistence(path)
        persistence.save({"trackers": {"good": {"provider": "anthropic"}}})

        class Unserializable:
            pass

        assert persistence.save({"trackers": Unserializable()}) is False
        assert persistence.load() == {"trackers": {"good": {"provider": "anthropic"}}}

    def test_failed_write_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "state.json.gz"

        class Unserializable:
            pass

        LineagePersistence(path).save({"trackers": Unserializable()})
        assert list(tmp_path.glob(".state-*.tmp")) == []

    def test_flush_skips_when_nothing_changed(self, tmp_path):
        store = _store()
        persistence = LineagePersistence(tmp_path / "state.json.gz")
        fingerprint = lambda: (  # noqa: E731 - mirrors the server wiring
            len(store._trackers),
            sum(t._turn_number for t in store._trackers.values()),
        )
        persistence._export = store.export_state
        persistence._fingerprint = fingerprint

        _served_turn(store, "sess-a", "anthropic", _messages("a"), _messages("a"))
        assert persistence.flush() is True
        assert persistence.flush() is False

        _served_turn(store, "sess-a", "anthropic", _messages("a", "b"), _messages("a", "b"))
        assert persistence.flush() is True

    def test_forced_flush_writes_even_when_unchanged(self, tmp_path):
        store = _store()
        persistence = LineagePersistence(tmp_path / "state.json.gz")
        persistence._export = store.export_state
        persistence._fingerprint = lambda: 0
        assert persistence.flush() is True
        assert persistence.flush() is False
        assert persistence.flush(force=True) is True

    def test_export_failure_does_not_raise(self, tmp_path):
        persistence = LineagePersistence(tmp_path / "state.json.gz")

        def boom():
            raise RuntimeError("export exploded")

        persistence._export = boom
        persistence._fingerprint = lambda: 1
        assert persistence.flush() is False

    def test_stop_without_start_is_safe(self, tmp_path):
        LineagePersistence(tmp_path / "state.json.gz").stop()


class TestEnableSwitch:
    def test_on_by_default(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_LINEAGE_PERSIST", raising=False)
        assert persistence_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", ""])
    def test_explicit_off(self, monkeypatch, value):
        monkeypatch.setenv("HEADROOM_LINEAGE_PERSIST", value)
        assert persistence_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_explicit_on(self, monkeypatch, value):
        monkeypatch.setenv("HEADROOM_LINEAGE_PERSIST", value)
        assert persistence_enabled() is True


class TestEndToEnd:
    def test_restart_preserves_the_replayable_prefix(self, tmp_path):
        """Full path: serve, snapshot to disk, restart, keep serving warm."""
        originals = _messages("turn one", "turn two")
        forwarded = _messages("turn one", "compressed turn two")

        store = _store()
        _served_turn(store, "sess-a", "anthropic", originals, forwarded)
        persistence = LineagePersistence(tmp_path / "state.json.gz")
        assert persistence.save(store.export_state())

        # Restart: brand new store, same file.
        snapshot = persistence.load()
        assert snapshot is not None
        revived = _store()
        assert revived.restore_state(snapshot) == 1

        extended = originals + _messages("turn three")
        tracker = revived.resolve_tracker("sess-a", "anthropic", extended)

        # The forwarded bytes overlay_cached_prefix needs are still here, so the
        # replay covers the whole recorded prefix instead of rewriting it.
        assert tracker.get_last_forwarded_messages() == forwarded
        assert tracker.get_frozen_message_count() > 0


def test_default_state_path_honours_dir_override(monkeypatch, tmp_path):
    """A test run must never write into the developer's own ~/.headroom.

    Regression guard. The default path is real user state, persistence defaults
    to on, and conftest scrubs HEADROOM_* which is exactly what exposes that
    default. A proxy built inside a test therefore wrote live conversation
    content into ~/.headroom/lineages/state.json.gz until the directory became
    overridable.
    """
    monkeypatch.setenv(DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    resolved = default_state_path()

    assert resolved.parent == tmp_path / "elsewhere"
    assert str(resolved).startswith(str(tmp_path))

    monkeypatch.delenv(DIR_ENV_VAR, raising=False)
    assert default_state_path().parent.name == "lineages"


def test_conftest_isolates_persistence_by_default():
    """The autouse fixture must be active for every test in this suite."""
    assert os.environ.get("HEADROOM_LINEAGE_PERSIST") == "0"
    assert not persistence_enabled()
    assert not str(default_state_path()).startswith(os.path.expanduser("~/.headroom"))


class TestLineageSnapshotFidelity:
    """The saved chain must be the object resolve_tracker compares against.

    Regression guard for the bug this feature shipped with. Lineages were
    rebuilt on load from each tracker's _last_original_messages rather than
    saved, on the assumption those equal what resolve_tracker stamps into
    family[key]. Against the live proxy they did not: the rebuilt chain never
    matched, every restart minted a fresh lineage, and the whole history was
    rewritten. Restart cost was unchanged, which is the entire point of the
    feature.
    """

    def test_saved_family_is_identical_to_the_live_one(self):
        store = _store()
        originals = _messages("first", "second")
        _served_turn(store, "sess-a", "anthropic", originals, originals)

        exported = store.export_state()
        assert exported["lineages"]["sess-a"] == store._lineages["sess-a"]

    def test_restored_family_is_identical_to_the_live_one(self):
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("a", "b"), _messages("a", "b"))

        revived = _store()
        revived.restore_state(store.export_state())
        assert revived._lineages["sess-a"] == store._lineages["sess-a"]

    def test_restart_does_not_mint_a_second_lineage(self):
        """The bug's live signature: one extra lineage per restart."""
        store = _store()
        originals = _messages("first", "second")
        _served_turn(store, "sess-a", "anthropic", originals, originals)
        assert len(store._trackers) == 1

        # Three restarts, each appending a turn as a real client would.
        history = originals
        current = store
        for extra in ("third", "fourth", "fifth"):
            revived = _store()
            revived.restore_state(current.export_state())
            history = history + _messages(extra)
            tracker = revived.resolve_tracker("sess-a", "anthropic", history)
            tracker.update_from_response(
                cache_read_tokens=1000,
                cache_write_tokens=100,
                messages=history,
                original_messages=history,
            )
            current = revived

        assert len(current._trackers) == 1, "a restart minted a fresh lineage"
        assert len(current._lineages["sess-a"]) == 1

    def test_lineages_for_dropped_trackers_are_not_saved(self):
        """An expired tracker is skipped, so its chain must not linger."""
        store = _store()
        tracker = _served_turn(
            store, "sess-a", "anthropic", _messages("a"), _messages("a")
        )
        tracker._last_activity = time.time() - 10_000

        exported = store.export_state()
        assert exported["trackers"] == {}
        assert exported["lineages"] == {}

    def test_load_tolerates_a_snapshot_with_no_lineages(self):
        """Older snapshots on disk predate the lineages key."""
        store = _store()
        _served_turn(store, "sess-a", "anthropic", _messages("a"), _messages("a"))
        legacy = store.export_state()
        del legacy["lineages"]

        revived = _store()
        assert revived.restore_state(legacy) == 1
        # Tracker state is still recovered; only the chain is absent.
        assert revived.get_or_create("sess-a", "anthropic")._turn_number > 0
        assert revived._lineages.get("sess-a", {}) == {}

    def test_survives_resolve_and_record_disagreeing(self):
        """The real production divergence, reproduced.

        resolve_tracker stamps family[key] from the request it is handed, while
        update_from_response records whatever the handler passes as
        original_messages. The two are not guaranteed to be the same object, and
        live traffic proves they are not. Rebuilding the chain from the recorded
        copy therefore produced a chain that the next request could not match.
        Saving family[key] itself has no such dependency.
        """
        store = _store()
        seen_by_resolve = _messages("first", "second")
        recorded_by_response = _messages("first", "second DIFFERENT")

        tracker = store.resolve_tracker("sess-a", "anthropic", seen_by_resolve)
        tracker.update_from_response(
            cache_read_tokens=1000,
            cache_write_tokens=100,
            messages=recorded_by_response,
            original_messages=recorded_by_response,
        )

        revived = _store()
        revived.restore_state(store.export_state())

        # The next request continues what resolve_tracker actually saw.
        extended = seen_by_resolve + _messages("third")
        revived.resolve_tracker("sess-a", "anthropic", extended)

        assert len(revived._trackers) == 1, "restart minted a fresh lineage"
