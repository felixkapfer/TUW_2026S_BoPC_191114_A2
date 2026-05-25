"""Command line interface for the option 2 helper commands."""

import argparse
import shutil
from pathlib import Path

from .helpers.commands import build_cpu_command, build_gpu_command
from .helpers.commands import run_cpu_command, run_gpu_command
from .helpers.paths import cuda_source_dir
from .utils.process import run_command


def positive_int(value: str) -> int:
    """Parse a positive integer from the command line."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all helper commands."""
    parser = argparse.ArgumentParser(description="BoPC option 2 helper")
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without changing anything")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Simple commands without additional options.
    subparsers.add_parser("check-env")
    subparsers.add_parser("build-gpu")
    subparsers.add_parser("build-cpu")

    # Run commands share most options.
    for name in ("run-gpu", "run-cpu"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--size", type=positive_int, default=1000)
        sub.add_argument("--nrep", type=positive_int, default=5)
        sub.add_argument("--output", type=Path)
        if name == "run-gpu":
            sub.add_argument("--block-x", type=positive_int)
            sub.add_argument("--block-y", type=positive_int)
    return parser


def print_env() -> int:
    """Show whether the needed build tools are available."""
    # nvcc is expected to be missing on many local laptops.
    for tool in ("make", "g++", "nvcc"):
        print(f"{tool}: {shutil.which(tool) or 'not found'}")
    return 0


def main() -> int:
    """Run the selected helper command."""
    args = make_parser().parse_args()
    src_dir = cuda_source_dir()

    if args.command == "check-env":
        return print_env()
    if args.command == "build-gpu":
        return run_command(build_gpu_command(), src_dir, args.dry_run)
    if args.command == "build-cpu":
        return run_command(build_cpu_command(), src_dir, args.dry_run)

    # Only GPU runs accept an optional CUDA block size.
    block = None
    if args.command == "run-gpu" and args.block_x and args.block_y:
        block = (args.block_x, args.block_y)

    command = run_gpu_command(args.size, args.nrep, block, args.output)
    if args.command == "run-cpu":
        command = run_cpu_command(args.size, args.nrep, args.output)
    return run_command(command, src_dir, args.dry_run)

