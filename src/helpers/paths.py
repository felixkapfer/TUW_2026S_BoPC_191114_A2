"""Path helpers for the assignment files."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    # paths.py is located at root/src/helpers/paths.py.
    current_file_path = Path(__file__).resolve()

    # Go up from helpers/ to src/ and then to the repository root.
    repository_root = current_file_path.parents[2]
    return repository_root


def cuda_source_dir() -> Path:
    """Return the committed CUDA source directory."""
    # task_description is ignored, so the buildable CUDA project lives in root.
    repository_root = project_root()

    # This is where the Makefiles and juliaset sources live.
    cuda_sources_directory = repository_root / "julia_cuda" / "src"
    return cuda_sources_directory
