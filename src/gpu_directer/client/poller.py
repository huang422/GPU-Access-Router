"""Poll the server queue until a request is complete."""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from gpu_directer.core.exceptions import GPUDirecterTimeoutError


def poll_for_result(
    base_url: str,
    request_id: str,
    timeout: int,
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """Poll GET {base_url}/gd/queue/{request_id} until complete.

    Returns Ollama-format ChatResponse dict on 'complete'.
    Raises GPUDirecterTimeoutError on 'timeout' status.
    Raises RuntimeError on 'error' status.
    """
    deadline = time.time() + timeout
    url = f"{base_url}/gd/queue/{request_id}"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"Failed to poll queue status: {exc}") from exc

        status = data.get("status", "")

        if status == "complete":
            result = data.get("result", {})
            return _reconstruct_chat_response(result)

        if status == "timeout":
            raise GPUDirecterTimeoutError(
                f"Request {request_id} exceeded queue timeout of {timeout}s"
            )

        if status == "error":
            raise RuntimeError(
                f"Server error for request {request_id}: {data.get('error', 'unknown error')}"
            )

        # Still waiting or processing — keep polling
        time.sleep(poll_interval)

    raise GPUDirecterTimeoutError(
        f"Request {request_id} exceeded client-side timeout of {timeout}s"
    )


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
