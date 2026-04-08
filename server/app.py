"""
server/app.py — OpenEnv multi-mode deployment entry point.

Exposes the CircuitSynth FastAPI application with a required main() function
so the deployment validator can:
  - Import it as  server.app:app
  - Call it as    server.app:main
  - Run it as     python server/app.py
"""

from __future__ import annotations

import os
import sys

# Make project root importable from any working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Re-export the FastAPI application
from server import app  # noqa: E402

__all__ = ["app", "main"]


def main() -> None:
    """Start the uvicorn server. Entry point for deployment runners."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
