"""GET /api/health  ->  confirms the engine loads on the deployment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_lib"))
import engine  # noqa: E402
import http_util  # noqa: E402

app = http_util.make_app(get_fn=engine.health)
application = app
