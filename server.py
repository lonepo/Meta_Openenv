"""
server.py — Proxy module for backward compatibility.
All logic has moved to server/app.py.
"""
from server.app import app, main

if __name__ == "__main__":
    main()
