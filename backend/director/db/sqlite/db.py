import json
import sqlite3
import time
import logging
import os

from typing import List, Optional

from director.constants import DBType
from director.db.base import BaseDB
from director.db.sqlite.initialize import initialize_sqlite

logger = logging.getLogger(__name__)


class SQLiteDB(BaseDB):
    def __init__(self, db_path: str = None):
        self.db_type = DBType.SQLITE
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "director.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=True)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        logger.info("Connected to SQLite DB...")

    def create_session(
        self, session_id: str, video_id: str, collection_id: str,
        created_at: int = None, updated_at: int = None,
        metadata: dict = {}, **kwargs,
    ) -> None:
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())
        self.cursor.execute(
            """
            INSERT OR IGNORE INTO sessions
                (session_id, video_id, collection_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, video_id, collection_id, created_at, updated_at, json.dumps(metadata)),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> dict:
        self.cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = self.cursor.fetchone()
        if row is None:
            return {}
        session = dict(row)
        session["metadata"] = json.loads(session["metadata"])
        return session

    def get_sessions(self) -> list:
        self.cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        sessions = [dict(r) for r in self.cursor.fetchall()]
        for session in sessions:
            session["metadata"] = json.loads(session["metadata"])
        return sessions

    def add_or_update_msg_to_conv(
        self, session_id: str, conv_id: str, msg_id: str, msg_type: str,
        agents: List[str], actions: List[str], content: List[dict], status: str = None,
        created_at: int = None, updated_at: int = None, metadata: dict = {}, **kwargs,
    ) -> None:
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO conversations
                (session_id, conv_id, msg_id, msg_type, agents, actions, content,
                 status, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, conv_id, msg_id, msg_type, json.dumps(agents),
                json.dumps(actions), json.dumps(content), status, created_at,
                updated_at, json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def get_conversations(self, session_id: str) -> list:
        self.cursor.execute(
            "SELECT * FROM conversations WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        conversations = []
        for row in self.cursor.fetchall():
            conv = dict(row)
            conv["agents"] = json.loads(conv["agents"])
            conv["actions"] = json.loads(conv["actions"])
            conv["content"] = json.loads(conv["content"])
            conv["metadata"] = json.loads(conv["metadata"])
            conversations.append(conv)
        return conversations

    def get_context_messages(self, session_id: str) -> list:
        self.cursor.execute(
            "SELECT context_data FROM context_messages WHERE session_id = ?",
            (session_id,),
        )
        result = self.cursor.fetchone()
        return json.loads(result[0]) if result else {}

    def add_or_update_context_msg(
        self, session_id: str, context_messages: list,
        created_at: int = None, updated_at: int = None,
        metadata: dict = {}, **kwargs,
    ) -> None:
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO context_messages
                (context_data, session_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (json.dumps(context_messages), session_id, created_at, updated_at, json.dumps(metadata)),
        )
        self.conn.commit()

    def compare_and_swap_context_msg(
        self,
        session_id: str,
        expected_context: Optional[dict],
        context_messages: dict,
    ) -> bool:
        """Atomically replace context only when the observed document is unchanged."""
        now = int(time.time())
        new_context = json.dumps(context_messages)

        if expected_context is not None:
            expected = json.dumps(expected_context)
            self.cursor.execute(
                """
                UPDATE context_messages
                SET context_data = ?, updated_at = ?
                WHERE session_id = ? AND context_data = ?
                """,
                (new_context, now, session_id, expected),
            )
            if self.cursor.rowcount > 0:
                self.conn.commit()
                return True
            if expected_context != {}:
                self.conn.commit()
                return False

        # An observed empty document may mean the row is absent. Compete to
        # create it without overwriting a peer that got there first.
        self.cursor.execute(
            """
            INSERT OR IGNORE INTO context_messages
                (context_data, session_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_context, session_id, now, now, json.dumps({})),
        )
        changed = self.cursor.rowcount > 0
        self.conn.commit()
        return changed

    def delete_conversation(self, session_id: str) -> bool:
        self.cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_context(self, session_id: str) -> bool:
        self.cursor.execute("DELETE FROM context_messages WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        self.delete_conversation(session_id)
        self.delete_context(session_id)
        self.cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()
        session_deleted = self.cursor.rowcount > 0
        return session_deleted, [] if session_deleted else ["session"]

    def health_check(self) -> bool:
        try:
            self.cursor.execute(
                """
                SELECT COUNT(name) FROM sqlite_master
                WHERE type='table'
                AND name IN ('sessions', 'conversations', 'context_messages');
                """
            )
            if self.cursor.fetchone()[0] < 3:
                logger.info("Tables not found. Initializing SQLite DB...")
                initialize_sqlite(self.db_path)
            return True
        except Exception as e:
            logger.exception(f"SQLite health check failed: {e}")
            return False

    def __del__(self):
        self.conn.close()
