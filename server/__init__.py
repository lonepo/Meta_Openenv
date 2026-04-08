"""
server/__init__.py — Re-exports app from server.app so that both
  'server:app'     (legacy uvicorn string)
  'server.app:app' (multi-mode deployment validator)
resolve to the same FastAPI application.
"""
from server.app import app, main  # noqa: F401

__all__ = ["app", "main"]
