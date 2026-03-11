from pathlib import Path

DEFAULT_PORT = 11434       # Ollama's port
DEFAULT_API_PORT = 8080    # GPU-Directer FastAPI server port
DEFAULT_TIMEOUT = 300
DEFAULT_QUEUE_DEPTH = 10
DEFAULT_ROUTING_MODE = "auto"
CONFIG_PATH = Path.home() / ".gpu-directer" / "config.toml"
