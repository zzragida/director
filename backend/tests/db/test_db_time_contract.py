import time
from pathlib import Path

from director.db.postgres.db import PostgresDB
from director.db.sqlite.db import SQLiteDB
from director.db.sqlite.initialize import initialize_sqlite


def test_sqlite_current_epoch_comes_from_database(tmp_path: Path):
    db_path = tmp_path / "director-time.db"
    initialize_sqlite(str(db_path))
    db = SQLiteDB(str(db_path))

    before = int(time.time()) - 2
    db_now = db.current_epoch()
    after = int(time.time()) + 2

    assert before <= db_now <= after


class FakeCursor:
    def __init__(self):
        self.calls = []
        self._row = {"epoch": 424242}

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchone(self):
        return self._row


class FakeConnection:
    def close(self):
        pass


def test_postgres_current_epoch_uses_database_clock_timestamp():
    db = PostgresDB.__new__(PostgresDB)
    db.cursor = FakeCursor()
    db.conn = FakeConnection()

    assert db.current_epoch() == 424242
    query, params = db.cursor.calls[0]
    assert "clock_timestamp()" in query
    assert "EXTRACT(EPOCH" in query
    assert params is None
