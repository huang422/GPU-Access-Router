from pathlib import Path

DEFAULT_PORT = 11434       # Ollama's port
DEFAULT_API_PORT = 9090    # GPU-Access-Router FastAPI server port
DEFAULT_TIMEOUT = 300
DEFAULT_QUEUE_DEPTH = 10
DEFAULT_ROUTING_MODE = "auto"
DEFAULT_FALLBACK_MODEL = ""
CONFIG_PATH = Path.home() / ".gpu-access-router" / "config.toml"
