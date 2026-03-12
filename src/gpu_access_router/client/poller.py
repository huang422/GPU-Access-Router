"""Response reconstruction helpers for remote inference results."""

from typing import Any, Dict


def _reconstruct_chat_response(result: Dict[str, Any]):
    """Reconstruct an ollama.ChatResponse from the server response dict."""
    try:
        import ollama

        msg_data = result.get("message", {})
        message = ollama.Message(
            role=msg_data.get("role", "assistant"),
            content=msg_data.get("content", ""),
        )
        # Build a ChatResponse object
        response = ollama.ChatResponse(
            model=result.get("model", ""),
            created_at=result.get("created_at", ""),
            message=message,
            done=result.get("done", True),
        )
        return response
    except Exception:
        # Fallback: return a simple namespace that mimics ChatResponse
        return _SimpleResponse(result)


class _SimpleResponse:
    """Minimal ChatResponse-compatible object for fallback."""

    def __init__(self, data: Dict[str, Any]):
        msg_data = data.get("message", {})
        self.model = data.get("model", "")
        self.created_at = data.get("created_at", "")
        self.done = data.get("done", True)
        self.message = _SimpleMessage(msg_data)

    def __repr__(self):
        return f"ChatResponse(model={self.model!r}, message={self.message!r})"


class _SimpleMessage:
    def __init__(self, data: Dict[str, Any]):
        self.role = data.get("role", "assistant")
        self.content = data.get("content", "")

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content!r})"
