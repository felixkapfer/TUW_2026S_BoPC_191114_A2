"""Command builders for building and running the CUDA code."""

from pathlib import Path
from typing import List, Optional, Tuple


def build_gpu_command() -> List[str]:
    """Return the command that builds the CUDA binary."""
    # -B rebuilds object files and avoids mixing CPU and GPU objects.
    # Makefile.gpu uses nvcc and produces juliaset_gpu.
    return ["make", "-B", "-f", "Makefile.gpu"]


def build_cpu_command() -> List[str]:
    """Return the command that builds the CPU reference binary."""
    # The CPU binary is useful for comparing timings against the GPU.
    # The plain Makefile uses g++ and produces juliaset_cpu.
    return ["make", "-B"]


def run_gpu_command(image_size: int, repetition_count: int,
                    cuda_block_size: Optional[Tuple[int, int]],
                    output_file: Optional[Path]) -> List[str]:
    """Return a juliaset_gpu command for one square image size."""
    # The assignment uses square images for all required measurements.
    # -r expects width and height, so pass the same value twice.
    gpu_command = [
            "./juliaset_gpu",
            "-r",
            str(image_size),
            str(image_size),
            "-n",
            str(repetition_count),
    ]

    # Block size is optional because kernel_gpu.cu has a default.
    if cuda_block_size is not None:
        # -b expects the CUDA block dimensions in x and y direction.
        gpu_command += ["-b", str(cuda_block_size[0]),
                        str(cuda_block_size[1])]

    # Writing CSV is optional; stdout is useful during quick tests.
    if output_file is not None:
        # -o writes the timing output into the given CSV file.
        gpu_command += ["-o", str(output_file)]

    return gpu_command


def run_cpu_command(image_size: int, repetition_count: int,
                    output_file: Optional[Path]) -> List[str]:
    """Return a juliaset_cpu command for one square image size."""
    # CPU and GPU runs use the same resolution and repetition arguments.
    cpu_command = [
            "./juliaset_cpu",
            "-r",
            str(image_size),
            str(image_size),
            "-n",
            str(repetition_count),
    ]

    # Keep stdout as the default when no CSV path was requested.
    if output_file is not None:
        cpu_command += ["-o", str(output_file)]

    return cpu_command
