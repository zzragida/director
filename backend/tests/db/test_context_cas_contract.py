from pathlib import Path

from director.db.postgres.db import PostgresDB
from director.db.sqlite.db import SQLiteDB
from director.db.sqlite.initialize import initialize_sqlite


def test_sqlite_context_cas_allows_only_one_first_writer(tmp_path: Path):
    db_path = tmp_path / "director-cas.db"
    initialize_sqlite(str(db_path))
    first = SQLiteDB(str(db_path))
    second = SQLiteDB(str(db_path))
    first.create_session("session-1", None, "collection-1")

    first_snapshot = first.get_context_messages("session-1")
    second_snapshot = second.get_context_messages("session-1")
    assert first_snapshot == {}
    assert second_snapshot == {}

    assert first.compare_and_swap_context_msg(
        "session-1",
        first_snapshot,
        {"lease": "worker-a"},
    ) is True
    assert second.compare_and_swap_context_msg(
        "session-1",
        second_snapshot,
        {"lease": "worker-b"},
    ) is False
    assert first.get_context_messages("session-1") == {"lease": "worker-a"}


def test_sqlite_context_cas_rejects_stale_existing_snapshot(tmp_path: Path):
    db_path = tmp_path / "director-cas-stale.db"
    initialize_sqlite(str(db_path))
    first = SQLiteDB(str(db_path))
    second = SQLiteDB(str(db_path))
    first.create_session("session-1", None, "collection-1")
    assert first.compare_and_swap_context_msg(
        "session-1", {}, {"revision": 1, "owner": "initial"}
    ) is True

    first_snapshot = first.get_context_messages("session-1")
    second_snapshot = second.get_context_messages("session-1")

    assert first.compare_and_swap_context_msg(
        "session-1",
        first_snapshot,
        {"revision": 2, "owner": "worker-a"},
    ) is True
    assert second.compare_and_swap_context_msg(
        "session-1",
        second_snapshot,
        {"revision": 2, "owner": "worker-b"},
    ) is False
    assert second.get_context_messages("session-1") == {
        "revision": 2,
        "owner": "worker-a",
    }


class FakeConnection:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


class FakeCursor:
    def __init__(self, rowcounts):
        self.rowcounts = list(rowcounts)
        self.rowcount = 0
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))
        self.rowcount = self.rowcounts.pop(0)


def make_postgres_db(rowcounts):
    db = PostgresDB.__new__(PostgresDB)
    db.cursor = FakeCursor(rowcounts)
    db.conn = FakeConnection()
    return db


def test_postgres_context_cas_uses_jsonb_equality_for_existing_document():
    db = make_postgres_db([1])

    changed = db.compare_and_swap_context_msg(
        "session-1",
        {"revision": 1},
        {"revision": 2},
    )

    assert changed is True
    query, params = db.cursor.calls[0]
    assert "context_data = %s::jsonb" in query
    assert params[2] == "session-1"
    assert db.conn.commits == 1


def test_postgres_empty_context_competes_with_insert_on_conflict():
    db = make_postgres_db([0, 1])

    changed = db.compare_and_swap_context_msg(
        "session-1",
        {},
        {"lease": "worker-a"},
    )

    assert changed is True
    assert len(db.cursor.calls) == 2
    assert "UPDATE context_messages" in db.cursor.calls[0][0]
    assert "ON CONFLICT (session_id) DO NOTHING" in db.cursor.calls[1][0]
    assert db.conn.commits == 1
