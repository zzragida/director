import copy

import pytest

from director.core.generation_lease import (
    GenerationLease,
    GenerationLeaseLostError,
    GenerationLeaseStore,
    safe_lease_summary,
)


class AtomicFakeDB:
    def __init__(self):
        self.context = {}
        self.fail_next_cas = False

    def get_context_messages(self, session_id):
        return copy.deepcopy(self.context.get(session_id, {}))

    def compare_and_swap_context_msg(self, session_id, expected_context, context_messages):
        if self.fail_next_cas:
            self.fail_next_cas = False
            current = copy.deepcopy(self.context.get(session_id, {}))
            current["peer"] = [{"role": "system", "content": "preserve-me"}]
            self.context[session_id] = current
            return False
        current = self.context.get(session_id, {})
        if current != expected_context:
            return False
        self.context[session_id] = copy.deepcopy(context_messages)
        return True


class FakeSession:
    def __init__(self, db, session_id="session-1"):
        self.db = db
        self.session_id = session_id


def make_store(db=None):
    db = db or AtomicFakeDB()
    return db, GenerationLeaseStore(FakeSession(db))


def test_only_one_owner_can_hold_an_active_operation_lease():
    _, store = make_store()

    first = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-a",
        ttl_seconds=60,
        now_epoch=1000,
    )
    second = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-b",
        ttl_seconds=60,
        now_epoch=1001,
    )

    assert first.acquired is True
    assert second.acquired is False
    assert second.reason_code == "operation_locked"
    assert second.lease.lease_token == first.lease.lease_token


def test_expired_lease_can_be_taken_over_with_a_new_token():
    _, store = make_store()
    first = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-a",
        ttl_seconds=30,
        now_epoch=1000,
    )
    takeover = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-b",
        ttl_seconds=30,
        now_epoch=1031,
    )

    assert takeover.acquired is True
    assert takeover.replaced_expired_lease is True
    assert takeover.reason_code == "expired_lease_replaced"
    assert takeover.lease.owner_id == "worker-b"
    assert takeover.lease.lease_token != first.lease.lease_token


def test_heartbeat_extends_only_the_current_matching_lease():
    _, store = make_store()
    result = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-a",
        ttl_seconds=30,
        now_epoch=1000,
    )

    refreshed = store.heartbeat(
        result.lease,
        ttl_seconds=30,
        now_epoch=1010,
    )
    assert refreshed.heartbeat_at_epoch == 1010
    assert refreshed.expires_at_epoch == 1040

    forged = result.lease.model_copy(deep=True)
    forged.lease_token = "not-the-current-token"
    with pytest.raises(GenerationLeaseLostError):
        store.heartbeat(forged, ttl_seconds=30, now_epoch=1011)


def test_expired_owner_cannot_heartbeat_after_takeover_window():
    _, store = make_store()
    result = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-a",
        ttl_seconds=30,
        now_epoch=1000,
    )

    with pytest.raises(GenerationLeaseLostError):
        store.heartbeat(result.lease, ttl_seconds=30, now_epoch=1030)


def test_matching_owner_can_release_and_next_owner_can_acquire():
    _, store = make_store()
    first = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-a",
        ttl_seconds=60,
        now_epoch=1000,
    )

    assert store.release(first.lease) is True
    assert store.get(first.lease.operation_id) is None

    second = store.acquire(
        first.lease.operation_id,
        owner_id="worker-b",
        ttl_seconds=60,
        now_epoch=1001,
    )
    assert second.acquired is True


def test_cas_retry_preserves_unrelated_peer_context():
    db = AtomicFakeDB()
    db.fail_next_cas = True
    store = GenerationLeaseStore(FakeSession(db))

    result = store.acquire(
        "genop:scene_video:0:test",
        owner_id="worker-a",
        ttl_seconds=60,
        now_epoch=1000,
    )

    assert result.acquired is True
    assert db.context["session-1"]["peer"][0]["content"] == "preserve-me"
    assert store.get(result.lease.operation_id) is not None


def test_safe_summary_does_not_expose_owner_or_token():
    lease = GenerationLease(
        operation_id="genop:scene_video:0:test",
        owner_id="worker-a:secret-owner",
        lease_token="secret-token",
        acquired_at_epoch=1000,
        heartbeat_at_epoch=1010,
        expires_at_epoch=1100,
    )

    summary = safe_lease_summary(lease, now_epoch=1020)

    assert summary["locked"] is True
    assert summary["operation_id"] == lease.operation_id
    assert "owner_id" not in summary
    assert "lease_token" not in summary
    assert "secret-token" not in str(summary)
    assert "secret-owner" not in str(summary)
