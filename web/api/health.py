"""GET /api/health  ->  confirms the engine loads on the deployment."""
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_lib"))
import engine  # noqa: E402
import http_util  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            http_util.send(self, 200, engine.health())
        except Exception as exc:
            http_util.send(self, 500, {"ok": False, "error": type(exc).__name__})
