"""Tiny subprocess wrapper with dry-run support."""

import shlex
import subprocess
from pathlib import Path


def printable_command(command: list[str], cwd: Path) -> str:
    """Return a shell-like command string for display."""
    # shlex.quote protects spaces and special characters in paths.
    parts = " ".join(shlex.quote(part) for part in command)
    return f"(cd {shlex.quote(str(cwd))} && {parts})"


def run_command(command: list[str], cwd: Path, dry_run: bool) -> int:
    """Print a command, then optionally execute it."""
    # Always print first so users can see exactly what would happen.
    print("$", printable_command(command, cwd), flush=True)

    # In dry-run mode we stop before touching the local machine.
    if dry_run:
        return 0

    subprocess.run(command, cwd=cwd, check=True)
    return 0

