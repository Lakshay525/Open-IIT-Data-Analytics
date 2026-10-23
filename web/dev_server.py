"""Local stand-in for Vercel: serves public/ and routes /api/* to the same handlers.

    python web/dev_server.py [port]      ->  http://127.0.0.1:8000
"""
import importlib
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public"
sys.path.insert(0, str(HERE / "api"))
sys.path.insert(0, str(HERE / "api" / "_lib"))


class Dev(BaseHTTPRequestHandler):
    def _api(self, verb):
        name = self.path.split("?")[0].rstrip("/").split("/")[-1]
        try:
            mod = importlib.import_module(name)
            getattr(mod.handler, f"do_{verb}")(self)
        except (ImportError, AttributeError):
            self.send_error(404, "no such endpoint")

    def do_POST(self):
        self._api("POST")

    def do_OPTIONS(self):
        self._api("POST")

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api("GET")
        rel = self.path.split("?")[0].lstrip("/") or "index.html"
        f = (PUBLIC / rel).resolve()
        if PUBLIC not in f.parents and f != PUBLIC:
            return self.send_error(403)
        if f.is_dir():
            f = f / "index.html"
        if not f.exists():
            return self.send_error(404)
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(f))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"live demo at http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Dev).serve_forever()
