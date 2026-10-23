"""Read-only paths for the serverless API (the project folder is read-only on Vercel)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA
RES = DATA
