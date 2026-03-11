"""Serial inference queue — one request at a time, prevents GPU OOM."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from gpu_directer.core.constants import DEFAULT_QUEUE_DEPTH, DEFAULT_TIMEOUT


@dataclass
class InferenceRequest:
    request_id: str
    model: str
    messages: list
    options: Dict[str, Any]
    timeout_seconds: int
    queued_at: float = field(default_factory=lambda: _now_ts())
    started_at: Optional[float] = None
    status: str = "waiting"          # waiting | processing | complete | timeout | error
    queue_position: int = 0
    result: Optional[Dict] = None
    error: Optional[str] = None


def _now_ts() -> float:
    import time
    return time.time()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SerialQueue:
    """Asyncio-based serial queue that processes one inference at a time."""

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT,
        max_depth: int = DEFAULT_QUEUE_DEPTH,
        ollama_callable: Optional[Callable] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth  # 0 = unlimited
        self._ollama_callable: Optional[Callable] = ollama_callable

        self._queue: asyncio.Queue = asyncio.Queue()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._pending: Dict[str, InferenceRequest] = {}
        self._task: Optional[asyncio.Task] = None
        self._started_at: float = _now_ts()

    def set_ollama_callable(self, callable_: Callable) -> None:
        self._ollama_callable = callable_

    def start(self) -> None:
        """Start the background processing loop (call from asyncio context)."""
        self._task = asyncio.create_task(self._process_loop())

    async def enqueue(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a request to the queue; returns {request_id, queue_position, status}."""
        req = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model=request_data["model"],
            messages=request_data.get("messages", []),
            options=request_data.get("options", {}),
            timeout_seconds=request_data.get("timeout", self.timeout_seconds),
        )
        position = self._queue.qsize() + 1
        req.queue_position = position
        self._pending[req.request_id] = req
        await self._queue.put(req)
        return {
            "request_id": req.request_id,
            "queue_position": position,
            "status": "waiting",
        }

    async def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Return current status dict for a request_id, or None if not found."""
        req = self._pending.get(request_id)
        if req is None:
            return None
        result: Dict[str, Any] = {
            "request_id": req.request_id,
            "status": req.status,
            "queue_position": req.queue_position,
            "queued_at": _fmt_ts(req.queued_at),
        }
        if req.started_at is not None:
            result["started_at"] = _fmt_ts(req.started_at)
        if req.status == "complete" and req.result is not None:
            result["result"] = req.result
        if req.status in ("timeout", "error") and req.error:
            result["error"] = req.error
            if req.status == "timeout":
                result["timeout_seconds"] = req.timeout_seconds
        return result

    async def get_depth(self) -> int:
        """Return total number of waiting + processing requests."""
        return len(self._pending)

    def get_uptime(self) -> float:
        return _now_ts() - self._started_at

    async def _process_loop(self) -> None:
        """Background task: pull from queue and run inference serially."""
        while True:
            req: InferenceRequest = await self._queue.get()
            if req.request_id not in self._pending:
                self._queue.task_done()
                continue
            async with self._lock:
                import time
                req.started_at = time.time()
                req.status = "processing"
                req.queue_position = 0
                # Update positions of remaining waiting requests
                self._recalculate_positions()

                try:
                    if self._ollama_callable is None:
                        raise RuntimeError("No Ollama callable configured.")
                    result = await asyncio.wait_for(
                        self._call_ollama(req),
                        timeout=req.timeout_seconds,
                    )
                    req.result = result
                    req.status = "complete"
                except asyncio.TimeoutError:
                    req.status = "timeout"
                    req.error = f"Request {req.request_id} exceeded queue timeout of {req.timeout_seconds}s"
                except Exception as exc:
                    req.status = "error"
                    req.error = str(exc)
                finally:
                    self._queue.task_done()

    async def _call_ollama(self, req: InferenceRequest) -> Dict:
        """Invoke the Ollama callable (may be sync or async)."""
        import inspect
        coro = self._ollama_callable(req.model, req.messages, req.options)
        if inspect.isawaitable(coro):
            return await coro
        return coro

    def _recalculate_positions(self) -> None:
        """Update queue_position for all waiting requests."""
        pos = 1
        for req in self._pending.values():
            if req.status == "waiting":
                req.queue_position = pos
                pos += 1


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
