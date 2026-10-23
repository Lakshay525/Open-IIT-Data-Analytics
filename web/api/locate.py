"""POST /api/locate  {"text": "...", "town": "T1"}  ->  pin, calibrated radii, tier, directions."""
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_lib"))
import engine  # noqa: E402
import http_util  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        http_util.post(self, lambda p: engine.locate(p["text"], p["town"]))

    def do_OPTIONS(self):
        http_util.send(self, 200, {})
