"""File naming and caching shared by the tools.

Tools pass *file paths* to each other rather than in-memory objects: this
keeps every tool independent (and callable from the command line, a notebook
or the agent), and every intermediate result can be opened in QGIS/CloudCompare.

"""

from __future__ import annotations

from pathlib import Path

from src import config


def derived_path(path: str | Path, suffix: str, extension: str | None = None, folder: Path | None = None) -> Path:
    path = Path(path)
    extension = extension or path.suffix
    return (folder or path.parent) / f"{path.stem}_{suffix}{extension}"


def is_up_to_date(output: Path, *inputs: Path) -> bool:
    """True if `output` exists and is newer than every input (like `make`)."""
    if not output.exists():
        return False
    return all(output.stat().st_mtime >= Path(i).stat().st_mtime for i in inputs if Path(i).exists())


def resolve_input(path: str) -> Path:
    """Accept absolute paths, paths relative to the project, or bare file names in data/processed."""
    candidates = [Path(path), config.PROJECT_ROOT / path, config.PROCESSED_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"File not found: {path}")


def relative(path: Path) -> str:
    """Short path for tool outputs (fewer tokens, easier to read)."""
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)
