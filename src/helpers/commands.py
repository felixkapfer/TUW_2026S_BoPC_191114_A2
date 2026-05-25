"""Command builders for building and running the CUDA code."""

from pathlib import Path


def build_gpu_command() -> list[str]:
    """Return the command that builds the CUDA binary."""
    # -B rebuilds object files and avoids mixing CPU and GPU objects.
    return ["make", "-B", "-f", "Makefile.gpu"]


def build_cpu_command() -> list[str]:
    """Return the command that builds the CPU reference binary."""
    # The CPU binary is useful for comparing timings against the GPU.
    return ["make", "-B"]


def run_gpu_command(size: int, nrep: int, block: tuple[int, int] | None,
                    output: Path | None) -> list[str]:
    """Return a juliaset_gpu command for one square image size."""
    # The assignment uses square images for all required measurements.
    command = ["./juliaset_gpu", "-r", str(size), str(size), "-n", str(nrep)]

    # Block size is optional because kernel_gpu.cu has a default.
    if block is not None:
        command += ["-b", str(block[0]), str(block[1])]

    # Writing CSV is optional; stdout is useful during quick tests.
    if output is not None:
        command += ["-o", str(output)]

    return command


def run_cpu_command(size: int, nrep: int, output: Path | None) -> list[str]:
    """Return a juliaset_cpu command for one square image size."""
    command = ["./juliaset_cpu", "-r", str(size), str(size), "-n", str(nrep)]

    if output is not None:
        command += ["-o", str(output)]

    return command

