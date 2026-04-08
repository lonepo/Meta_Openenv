"""
server/app.py — OpenEnv-compliant FastAPI entry point.

This module re-exports the FastAPI `app` from the root server.py so that
the multi-mode deployment validator can find it at `server.app:app`.
"""

import sys
import os

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # re-export the FastAPI application

__all__ = ["app"]
