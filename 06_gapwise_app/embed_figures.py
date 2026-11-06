"""Attach generated figures to executed notebooks without recomputing analysis."""
import base64
import json
from pathlib import Path
import nbformat
from data import OUTPUT


def embed(folder):
    path = folder / "analysis_executed.ipynb"
    manifest = json.loads((folder / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only complete runs can receive a figure gallery")
    notebook = nbformat.read(path, as_version=4)
    # Idempotent refresh: replace only cells explicitly owned by this gallery.
    notebook.cells = [c for c in notebook.cells if not c.metadata.get("v4_figure_gallery")]
    title = nbformat.v4.new_markdown_cell("## Generated analysis figures\n\nFigures from this exact scenario run, in generation order. Numerical tables are exported alongside the notebook.")
    title.metadata["v4_figure_gallery"] = True
    notebook.cells.append(title)
    for name in manifest["figures"]:
        encoded = base64.b64encode((folder / "figures" / name).read_bytes()).decode()
        cell = nbformat.v4.new_markdown_cell(f"![{name}](attachment:{name})")
        cell.attachments = {name: {"image/png": encoded}}
        cell.metadata["v4_figure_gallery"] = True
        notebook.cells.append(cell)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return len(manifest["figures"])


if __name__ == "__main__":
    for folder in (OUTPUT / "metrics").iterdir():
        if folder.is_dir() and (folder / "analysis_executed.ipynb").exists():
            print(folder.name, embed(folder))
