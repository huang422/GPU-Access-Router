"""Client-side components for GPU Access Router."""

try:
    import ollama  # noqa: F401
    import httpx  # noqa: F401
    HAVE_CLIENT = True
except ImportError:
    HAVE_CLIENT = False
