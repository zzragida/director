import pytest

from director.db.postgres.db import PostgresDB
from director.db.sqlite.db import SQLiteDB
from director.db.sqlite.initialize import initialize_sqlite


class FakeConnection:
    def __init__(self):
        self.commits = 0
        self.closed = False

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, rowcount):
        self.rowcount = rowcount
        self.executed = []

    def execute(self, query, params):
        self.executed.append((query, params))


def _sqlite_db(tmp_path):
    db_path = tmp_path / "director-test.db"
    initialize_sqlite(str(db_path))
    return SQLiteDB(str(db_path))


def test_sqlite_delete_session_with_related_rows(tmp_path):
    db = _sqlite_db(tmp_path)
    session_id = "session-with-related-data"

    db.create_session(session_id, video_id="video-1", collection_id="collection-1")
    db.add_or_update_msg_to_conv(
        session_id=session_id,
        conv_id="conv-1",
        msg_id="msg-1",
        msg_type="input",
        agents=[],
        actions=[],
        content=[],
        status="success",
    )
    db.add_or_update_context_msg(session_id, {"reasoning": []})

    success, failed_components = db.delete_session(session_id)

    assert success is True
    assert failed_components == []
    assert db.get_session(session_id) == {}
    assert db.get_conversations(session_id) == []
    assert db.get_context_messages(session_id) == {}


def test_sqlite_delete_session_without_optional_rows_is_success(tmp_path):
    db = _sqlite_db(tmp_path)
    session_id = "session-only"
    db.create_session(session_id, video_id=None, collection_id="collection-1")

    success, failed_components = db.delete_session(session_id)

    assert success is True
    assert failed_components == []
    assert db.get_session(session_id) == {}


def test_sqlite_delete_missing_session_is_failure(tmp_path):
    db = _sqlite_db(tmp_path)

    success, failed_components = db.delete_session("missing-session")

    assert success is False
    assert failed_components == ["session"]


def _postgres_db(session_rowcount, child_delete_result=False):
    db = PostgresDB.__new__(PostgresDB)
    db.conn = FakeConnection()
    db.cursor = FakeCursor(rowcount=session_rowcount)
    db.delete_conversation = lambda session_id: child_delete_result
    db.delete_context = lambda session_id: child_delete_result
    return db


def test_postgres_delete_session_without_optional_rows_is_success():
    db = _postgres_db(session_rowcount=1, child_delete_result=False)

    success, failed_components = db.delete_session("session-only")

    assert success is True
    assert failed_components == []
    assert db.conn.commits == 1
    assert db.cursor.executed == [
        ("DELETE FROM sessions WHERE session_id = %s", ("session-only",))
    ]


def test_postgres_delete_missing_session_is_failure():
    db = _postgres_db(session_rowcount=0, child_delete_result=False)

    success, failed_components = db.delete_session("missing-session")

    assert success is False
    assert failed_components == ["session"]


def test_postgres_delete_propagates_database_errors():
    db = _postgres_db(session_rowcount=1)

    def fail_delete(_session_id):
        raise RuntimeError("database unavailable")

    db.delete_context = fail_delete

    with pytest.raises(RuntimeError, match="database unavailable"):
        db.delete_session("session-1")
