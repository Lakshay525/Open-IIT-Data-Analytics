"""Single place for every folder the code reads or writes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                # supplied inputs (cleaned copies)
OUT = ROOT / "outputs"              # deliverables: pins, predictions, demo, offline packs
RES = OUT / "results"               # diagnostics: score tables, calibration tables, figures

for _p in (OUT, RES):
    _p.mkdir(parents=True, exist_ok=True)
