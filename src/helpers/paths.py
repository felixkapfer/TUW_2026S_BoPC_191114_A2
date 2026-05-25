"""Path helpers for the assignment files."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    # paths.py is located at root/src/helpers/paths.py.
    return Path(__file__).resolve().parents[2]


def cuda_source_dir() -> Path:
    """Return the committed CUDA source directory."""
    # task_description is ignored, so the buildable CUDA project lives in root.
    return project_root() / "julia_cuda" / "src"
