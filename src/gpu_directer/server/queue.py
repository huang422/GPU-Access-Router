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
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


def _now_ts() -> float:
    import time
    return time.time()


class SerialQueue:
    """Asyncio-based serial queue that processes one inference at a time."""

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT,
        max_depth: int = DEFAULT_QUEUE_DEPTH,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth  # 0 = unlimited
        self._ollama_callable: Optional[Callable] = None
        self._unload_callable: Optional[Callable] = None
        self._current_model: Optional[str] = None

        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending: Dict[str, InferenceRequest] = {}
        self._task: Optional[asyncio.Task] = None
        self._started_at: float = _now_ts()

    def set_ollama_callable(self, callable_: Callable) -> None:
        self._ollama_callable = callable_

    def set_unload_callable(self, callable_: Callable) -> None:
        self._unload_callable = callable_

    def start(self) -> None:
        """Start the background processing loop (call from async context)."""
        self._task = asyncio.create_task(self._process_loop())

    async def enqueue(self, request_data: Dict[str, Any]) -> "InferenceRequest":
        """Add a request to the queue and return the InferenceRequest object."""
        req = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model=request_data["model"],
            messages=request_data.get("messages", []),
            options=request_data.get("options", {}),
            timeout_seconds=request_data.get("timeout", self.timeout_seconds),
        )
        req.queue_position = self._queue.qsize() + 1
        self._pending[req.request_id] = req
        await self._queue.put(req)
        return req

    async def get_depth(self) -> int:
        """Return number of active (waiting + processing) requests."""
        return sum(
            1 for r in self._pending.values()
            if r.status in ("waiting", "processing")
        )

    def get_uptime(self) -> float:
        return _now_ts() - self._started_at

    async def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Return a status snapshot for the given request_id."""
        req = self._pending.get(request_id)
        if req is None:
            return None
        out: Dict[str, Any] = {
            "request_id": req.request_id,
            "status": req.status,
            "queue_position": req.queue_position,
            "queued_at": _fmt_ts(req.queued_at),
        }
        if req.started_at is not None:
            out["started_at"] = _fmt_ts(req.started_at)
        if req.status == "complete" and req.result is not None:
            out["result"] = req.result
        if req.status in ("timeout", "error") and req.error:
            out["error"] = req.error
        return out

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Pull requests from queue and run inference one at a time."""
        while True:
            req: InferenceRequest = await self._queue.get()
            if req.request_id not in self._pending:
                self._queue.task_done()
                continue

            import time
            req.started_at = time.time()
            req.status = "processing"
            req.queue_position = 0
            self._recalculate_positions()

            try:
                if self._ollama_callable is None:
                    raise RuntimeError("No Ollama callable configured.")

                # Unload previous model from GPU if switching models
                if (
                    self._unload_callable is not None
                    and self._current_model is not None
                    and self._current_model != req.model
                ):
                    try:
                        await self._run_in_thread(self._unload_callable, self._current_model)
                    except Exception:
                        pass  # best-effort

                result = await asyncio.wait_for(
                    self._run_in_thread(
                        self._ollama_callable, req.model, req.messages, req.options
                    ),
                    timeout=req.timeout_seconds,
                )
                self._current_model = req.model
                req.result = result
                req.status = "complete"

            except asyncio.TimeoutError:
                req.status = "timeout"
                req.error = f"Inference timed out after {req.timeout_seconds}s"
            except Exception as exc:
                req.status = "error"
                req.error = str(exc)
            finally:
                req.done_event.set()          # wake up any waiters
                self._queue.task_done()
                asyncio.create_task(self._cleanup_after(req.request_id, delay=300))

    @staticmethod
    async def _run_in_thread(fn: Callable, *args) -> Any:
        """Run a sync callable in a thread pool so the event loop stays free."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)

    async def _cleanup_after(self, request_id: str, delay: int) -> None:
        await asyncio.sleep(delay)
        self._pending.pop(request_id, None)

    def _recalculate_positions(self) -> None:
        pos = 1
        for r in self._pending.values():
            if r.status == "waiting":
                r.queue_position = pos
                pos += 1


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
