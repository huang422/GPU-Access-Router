try:
    import ollama  # noqa: F401
    HAVE_CLIENT = True
except ImportError:
    HAVE_CLIENT = False
