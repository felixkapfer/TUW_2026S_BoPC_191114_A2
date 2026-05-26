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
    # argparse gives us strings, so convert first and validate afterwards.
    parsed_integer = int(value)
    if parsed_integer <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed_integer


def make_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all helper commands."""
    # This top-level parser owns the shared flags and all subcommands.
    argument_parser = argparse.ArgumentParser(
            description="BoPC option 2 helper")
    argument_parser.add_argument("--dry-run", action="store_true",
                                 help="print actions without changing anything")

    # Every user action is modeled as a required subcommand.
    subcommand_parsers = argument_parser.add_subparsers(
            dest="command", required=True)

    # Simple commands without additional options.
    subcommand_parsers.add_parser("check-env")
    subcommand_parsers.add_parser("build-gpu")
    subcommand_parsers.add_parser("build-cpu")

    # Run commands share most options.
    for command_name in ("run-gpu", "run-cpu"):
        # Each run command gets its own parser but uses the same basic options.
        run_command_parser = subcommand_parsers.add_parser(command_name)
        run_command_parser.add_argument("--size", type=positive_int,
                                        default=1000)
        run_command_parser.add_argument("--nrep", type=positive_int,
                                        default=5)
        run_command_parser.add_argument("--output", type=Path)

        # CUDA block size only makes sense for GPU runs.
        if command_name == "run-gpu":
            run_command_parser.add_argument("--block-x", type=positive_int)
            run_command_parser.add_argument("--block-y", type=positive_int)
    return argument_parser


def print_env() -> int:
    """Show whether the needed build tools are available."""
    # nvcc is expected to be missing on many local laptops.
    for tool_name in ("make", "g++", "nvcc"):
        # shutil.which tells us whether the executable is on PATH.
        tool_path = shutil.which(tool_name)
        print(f"{tool_name}: {tool_path or 'not found'}")
    return 0


def main() -> int:
    """Run the selected helper command."""
    # Parse the command line once at the boundary of the program.
    parsed_arguments = make_parser().parse_args()

    # All build and run commands execute inside the CUDA source directory.
    cuda_sources_directory = cuda_source_dir()

    # Environment checks only print tool availability and do not run make.
    if parsed_arguments.command == "check-env":
        return print_env()

    # Build commands are direct wrappers around the project Makefiles.
    if parsed_arguments.command == "build-gpu":
        return run_command(build_gpu_command(), cuda_sources_directory,
                           parsed_arguments.dry_run)
    if parsed_arguments.command == "build-cpu":
        return run_command(build_cpu_command(), cuda_sources_directory,
                           parsed_arguments.dry_run)

    # Only GPU runs accept an optional CUDA block size.
    cuda_block_size = None
    if (parsed_arguments.command == "run-gpu"
            and parsed_arguments.block_x
            and parsed_arguments.block_y):
        cuda_block_size = (parsed_arguments.block_x, parsed_arguments.block_y)

    # GPU is the default run path here; CPU replaces it below if requested.
    helper_command = run_gpu_command(parsed_arguments.size,
                                     parsed_arguments.nrep,
                                     cuda_block_size,
                                     parsed_arguments.output)
    if parsed_arguments.command == "run-cpu":
        helper_command = run_cpu_command(parsed_arguments.size,
                                         parsed_arguments.nrep,
                                         parsed_arguments.output)

    # Finally execute the selected helper command, or print it for dry-runs.
    return run_command(helper_command, cuda_sources_directory,
                       parsed_arguments.dry_run)
