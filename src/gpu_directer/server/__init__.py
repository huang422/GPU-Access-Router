try:
    import fastapi  # noqa: F401
    HAVE_SERVER = True
except ImportError:
    HAVE_SERVER = False
