from copy import deepcopy
from typing import Callable, Dict, Optional


class ContextCASConflictError(RuntimeError):
    """Raised when a context document keeps changing during a CAS update."""


ContextMutator = Callable[[Dict], Optional[Dict]]


def atomic_update_context(
    db,
    session_id: str,
    mutator: ContextMutator,
    *,
    max_attempts: int = 8,
) -> Dict:
    """Apply a document mutation without overwriting a concurrent writer.

    The mutator receives an isolated copy of the most recently observed context
    document and must return the desired replacement document. Returning
    ``None`` means no write is required. The database implementation performs
    the compare-and-swap atomically.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    for _ in range(max_attempts):
        current = db.get_context_messages(session_id) or {}
        updated = mutator(deepcopy(current))
        if updated is None:
            return current
        if db.compare_and_swap_context_msg(session_id, current, updated):
            return updated

    raise ContextCASConflictError(
        f"Context changed concurrently for session {session_id!r}"
    )
