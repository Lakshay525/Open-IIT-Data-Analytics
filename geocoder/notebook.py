"""Build and execute notebooks/analysis.ipynb from the percent-format source notebooks/analysis.py.

Run after the pipeline:  python -m geocoder.notebook
"""
from __future__ import annotations

import re

import nbformat as nbf
from nbclient import NotebookClient

from .paths import ROOT

NL = chr(10)


def parse(text: str):
    """Split a '# %%' percent-format script into notebook cells."""
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        head, _, body = block.partition(NL)
        if "[markdown]" in head:
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip(NL).split(NL)]
            cells.append(nbf.v4.new_markdown_cell(NL.join(lines).strip()))
        else:
            cells.append(nbf.v4.new_code_cell(body.strip(NL)))
    return cells


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = parse((ROOT / "notebooks" / "analysis.py").read_text(encoding="utf-8"))
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    out = ROOT / "notebooks" / "analysis.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
