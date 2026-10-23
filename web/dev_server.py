"""Local stand-in for Vercel: serves public/ and runs the same WSGI apps from api/.

    python web/dev_server.py [port]      ->  http://127.0.0.1:8000
"""
import importlib
import mimetypes
import sys
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public"
sys.path.insert(0, str(HERE / "api"))
sys.path.insert(0, str(HERE / "api" / "_lib"))


def application(environ, start_response):
    path = environ["PATH_INFO"]
    if path.startswith("/api/"):
        name = path.rstrip("/").split("/")[-1]
        try:
            return importlib.import_module(name).app(environ, start_response)
        except ImportError:
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"no such endpoint"]
    f = (PUBLIC / (path.lstrip("/") or "index.html")).resolve()
    if f.is_dir():
        f = f / "index.html"
    if PUBLIC not in f.parents or not f.exists():
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]
    data = f.read_bytes()
    start_response("200 OK", [("Content-Type", mimetypes.guess_type(str(f))[0] or "application/octet-stream"), ("Content-Length", str(len(data)))])
    return [data]


class Quiet(WSGIRequestHandler):
    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"live demo at http://127.0.0.1:{port}")
    make_server("127.0.0.1", port, application, handler_class=Quiet).serve_forever()
