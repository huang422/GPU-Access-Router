"""FastAPI server exposing /gd/* endpoints with serial queue."""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Install with: pip install gpu-directer[server]"
    ) from exc

try:
    import ollama as _ollama
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Install with: pip install gpu-directer[server]"
    ) from exc

from gpu_directer.core.constants import DEFAULT_PORT, DEFAULT_QUEUE_DEPTH, DEFAULT_TIMEOUT
from gpu_directer.server.queue import SerialQueue

app = FastAPI(title="GPU Directer Server", version="0.1.0")

_serial_queue: Optional[SerialQueue] = None
_start_time: float = time.time()
_ollama_port: int = DEFAULT_PORT


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    options: Dict[str, Any] = {}
    timeout: int = DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _serial_queue, _ollama_port
    from gpu_directer import config as cfg_mod
    cfg = cfg_mod.load_config()
    _ollama_port = cfg.get("server", {}).get("ollama_port", DEFAULT_PORT)
    queue_timeout = cfg.get("server", {}).get("queue_timeout", DEFAULT_TIMEOUT)
    max_depth = cfg.get("server", {}).get("max_queue_depth", DEFAULT_QUEUE_DEPTH)

    _serial_queue = SerialQueue(timeout_seconds=queue_timeout, max_depth=max_depth)

    # Wire Ollama callable
    ollama_client = _ollama.Client(host=f"http://localhost:{_ollama_port}")

    def _ollama_call(model, messages, options):
        resp = ollama_client.chat(model=model, messages=messages, options=options or {})
        # Convert to dict for storage
        if hasattr(resp, "__dict__"):
            return _chat_response_to_dict(resp)
        return resp

    _serial_queue.set_ollama_callable(_ollama_call)
    _serial_queue.start()


def _chat_response_to_dict(resp) -> Dict:
    """Convert ollama.ChatResponse to a plain dict for JSON storage."""
    msg = resp.message
    return {
        "model": getattr(resp, "model", ""),
        "created_at": str(getattr(resp, "created_at", "")),
        "message": {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", ""),
        },
        "done": getattr(resp, "done", True),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/gd/health")
async def health():
    global _serial_queue, _ollama_port
    queue_depth = await _serial_queue.get_depth() if _serial_queue else 0
    processing = any(
        req.status == "processing"
        for req in (_serial_queue._pending.values() if _serial_queue else [])
    )
    ollama_ok = _check_ollama_reachable(_ollama_port)
    gpu_ok = _check_gpu_available()
    uptime = _serial_queue.get_uptime() if _serial_queue else 0.0
    return {
        "status": "ok" if ollama_ok else "degraded",
        "queue_depth": queue_depth,
        "processing": processing,
        "ollama_reachable": ollama_ok,
        "gpu_available": gpu_ok,
        "uptime_seconds": int(uptime),
    }


@app.post("/gd/chat", status_code=202)
async def submit_chat(req: ChatRequest):
    global _serial_queue
    if _serial_queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialized.")

    # Check queue full
    depth = await _serial_queue.get_depth()
    max_depth = _serial_queue.max_depth
    if max_depth > 0 and depth >= max_depth:
        return JSONResponse(
            status_code=503,
            content={"error": "Queue full", "max_depth": max_depth, "current_depth": depth},
        )

    result = await _serial_queue.enqueue({
        "model": req.model,
        "messages": req.messages,
        "options": req.options,
        "timeout": req.timeout,
    })
    return result


@app.get("/gd/queue/{request_id}")
async def get_queue_status(request_id: str):
    global _serial_queue
    if _serial_queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialized.")
    status = await _serial_queue.get_status(request_id)
    if status is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Request not found", "request_id": request_id},
        )
    return status


@app.get("/gd/queue")
async def get_queue():
    global _serial_queue
    if _serial_queue is None:
        return {"depth": 0, "max_depth": 0, "processing": False, "requests": []}
    depth = await _serial_queue.get_depth()
    processing = any(
        req.status == "processing"
        for req in _serial_queue._pending.values()
    )
    requests = [
        {
            "request_id": req.request_id,
            "position": req.queue_position,
            "model": req.model,
            "queued_at": _fmt_ts(req.queued_at),
        }
        for req in _serial_queue._pending.values()
        if req.status == "waiting"
    ]
    return {
        "depth": depth,
        "max_depth": _serial_queue.max_depth,
        "processing": processing,
        "requests": requests,
    }


@app.get("/gd/models")
async def list_models():
    global _ollama_port
    url = f"http://localhost:{_ollama_port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": f"Ollama unreachable: {exc}", "models": []},
        )
    return {"models": data.get("models", [])}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ollama_reachable(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def _check_gpu_available() -> bool:
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "exec", "ollama", "nvidia-smi"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _fmt_ts(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_server(host: str = "0.0.0.0", port: int = DEFAULT_PORT, reload: bool = False):
    """Start the FastAPI server via uvicorn."""
    uvicorn.run(
        "gpu_directer.server.api:app",
        host=host,
        port=port,
        reload=reload,
    )
