"""WSGI helpers shared by the serverless functions (the Vercel Python runtime expects a WSGI/ASGI app named `app`)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402

MAX_BODY = 20_000
HEADERS = [("Content-Type", "application/json; charset=utf-8"), ("Cache-Control", "no-store"),
           ("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "Content-Type"),
           ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")]
STATUS = {200: "200 OK", 400: "400 Bad Request", 404: "404 Not Found", 405: "405 Method Not Allowed", 500: "500 Internal Server Error"}


def _reply(start_response, code, obj):
    body = engine.dumps(obj)
    start_response(STATUS[code], HEADERS + [("Content-Length", str(len(body)))])
    return [body]


def make_app(post_fn=None, get_fn=None):
    """Build a WSGI app: GET calls get_fn(), POST calls post_fn(json_payload)."""
    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        try:
            if method == "OPTIONS":
                return _reply(start_response, 200, {})
            if method == "GET" and get_fn:
                return _reply(start_response, 200, get_fn())
            if method == "POST" and post_fn:
                n = int(environ.get("CONTENT_LENGTH") or 0)
                if n <= 0 or n > MAX_BODY:
                    return _reply(start_response, 400, {"error": "empty or oversized request"})
                payload = json.loads(environ["wsgi.input"].read(n).decode("utf-8"))
                result = post_fn(payload)
                return _reply(start_response, 400 if "error" in result else 200, result)
            return _reply(start_response, 405, {"error": "method not allowed"})
        except (ValueError, KeyError, TypeError) as exc:
            return _reply(start_response, 400, {"error": f"bad request: {exc}"})
        except Exception as exc:  # never leak a traceback to the browser
            return _reply(start_response, 500, {"error": f"server error: {type(exc).__name__}"})
    return app
