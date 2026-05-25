"""Root entry point for the BoPC option 2 helper commands."""

from src.cli import main


if __name__ == "__main__":
    # Keeping this file in the project root makes `uv run python main.py` work.
    main()

