"""Root entry point for the BoPC option 2 helper commands."""

# Import the real CLI entry point from the helper package.
from src.cli import main as run_cli


if __name__ == "__main__":
    # Only run the CLI when this file is executed directly.
    # Keeping this file in the project root makes `uv run python main.py` work.
    raise SystemExit(run_cli())
