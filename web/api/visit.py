"""POST /api/visit  {"address_id": "AD000123", "x": 1200.5, "y": -300.0}  ->  how the neighbours' pins move."""
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_lib"))
import engine  # noqa: E402
import http_util  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        http_util.post(self, lambda p: engine.what_if_visit(str(p["address_id"]), float(p["x"]), float(p["y"])))

    def do_OPTIONS(self):
        http_util.send(self, 200, {})
