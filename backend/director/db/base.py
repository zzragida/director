from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseDB(ABC):
    """Interface for all databases. It provides a common interface for all databases to follow."""

    @abstractmethod
    def create_session(
        self, session_id: str, video_id: str = None, collection_id: str = None
    ) -> None:
        """Create a new session."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> dict:
        """Get a session by session_id."""
        pass

    @abstractmethod
    def get_sessions(self) -> list:
        """Get all sessions."""
        pass

    @abstractmethod
    def add_or_update_msg_to_conv() -> None:
        """Add a new message (input or output) to the conversation."""
        pass

    @abstractmethod
    def get_conversations(self, session_id: str) -> list:
        """Get all conversations for a given session."""
        pass

    @abstractmethod
    def get_context_messages(self, session_id: str) -> list:
        """Get context messages for a session."""
        pass

    @abstractmethod
    def add_or_update_context_msg(
        self, session_id: str, context_messages: list
    ) -> None:
        """Update context messages for a session."""
        pass

    @abstractmethod
    def compare_and_swap_context_msg(
        self,
        session_id: str,
        expected_context: Optional[dict],
        context_messages: dict,
    ) -> bool:
        """Atomically replace a session context document when it still matches.

        ``expected_context`` is the document previously read by the caller. When
        it is ``None``, the method succeeds only if no context row exists yet.
        Return ``True`` only when this caller created/replaced the row. A
        concurrent writer must cause ``False`` rather than silently overwriting
        its update. Database errors should propagate.
        """
        pass

    def current_epoch(self) -> int:
        """Return authoritative database-server epoch seconds.

        Production database implementations must override this method using the
        database server clock. Keeping the contract on ``BaseDB`` lets lease
        code depend on one clock source instead of application worker clocks.
        """
        raise NotImplementedError("database server time is not implemented")

    @abstractmethod
    def delete_session(self, session_id: str) -> Tuple[bool, List[str]]:
        """Delete a session and its optional associated data.

        Return ``(True, [])`` only when the session row itself is deleted.
        Missing optional conversation/context rows are not failures. If no
        session row is deleted, return ``(False, ["session"])``. Database
        errors should propagate to the caller instead of being converted into
        a successful result.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the database is healthy."""
        pass
