"""Tiny subprocess wrapper with dry-run support."""

import shlex
import subprocess
from pathlib import Path
from typing import List


def printable_command(command: List[str], working_directory: Path) -> str:
    """Return a shell-like command string for display."""
    # shlex.quote protects spaces and special characters in paths.
    quoted_command_parts = " ".join(shlex.quote(part) for part in command)

    # Show the directory switch together with the command that will run there.
    quoted_working_directory = shlex.quote(str(working_directory))
    return f"(cd {quoted_working_directory} && {quoted_command_parts})"


def run_command(command: List[str], working_directory: Path,
                dry_run: bool) -> int:
    """Print a command, then optionally execute it."""
    # Always print first so users can see exactly what would happen.
    command_preview = printable_command(command, working_directory)
    print("$", command_preview, flush=True)

    # In dry-run mode we stop before touching the local machine.
    if dry_run:
        return 0

    # check=True turns failed commands into Python exceptions.
    subprocess.run(command, cwd=working_directory, check=True)
    return 0
