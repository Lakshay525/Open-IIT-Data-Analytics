"""Tiny helpers shared by the serverless handlers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402

MAX_BODY = 20_000


def send(h, code, obj):
    body = engine.dumps(obj)
    h.send_response(code)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Cache-Control", "no-store")
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def read_json(h):
    n = int(h.headers.get("Content-Length") or 0)
    if n <= 0 or n > MAX_BODY:
        raise ValueError("empty or oversized request")
    return json.loads(h.rfile.read(n).decode("utf-8"))


def post(h, fn):
    try:
        payload = read_json(h)
        result = fn(payload)
        send(h, 400 if "error" in result else 200, result)
    except (ValueError, KeyError, TypeError) as exc:
        send(h, 400, {"error": f"bad request: {exc}"})
    except Exception as exc:  # never leak a traceback to the browser
        send(h, 500, {"error": f"server error: {type(exc).__name__}"})
