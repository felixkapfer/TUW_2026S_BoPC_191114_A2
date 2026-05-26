#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "kernel.hpp"

extern int global_block_x, global_block_y;

namespace {
void checkCuda(cudaError_t result, const char *step) {
    // Stop early if a CUDA call failed.
    if(result != cudaSuccess) {
        std::cerr << "CUDA error during " << step << ": "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__device__ int countJuliaIterations(int pixel_x, int pixel_y,
        Complex julia_constant, float zoom_scale, int image_width,
        int image_height, int max_iterations, float max_magnitude,
        float horizontal_scale, float vertical_scale) {
    // Move the pixel coordinate into the complex plane.
    // The image center becomes the origin.
    float scaled_x = zoom_scale * horizontal_scale
            * (float)(pixel_x - image_width / 2) / (image_width / 2);

    // Scale y separately so non-square images keep the right proportions.
    float scaled_y = zoom_scale * vertical_scale
            * (float)(pixel_y - image_height / 2) / (image_height / 2);

    // This is the point we keep squaring for the Julia iteration.
    Complex current_point(scaled_x, scaled_y);

    // Count how long the point stays inside the allowed magnitude.
    int iteration = 0;
    for(iteration = 0; iteration < max_iterations; iteration++) {
        // Julia step: z = z^2 + c.
        current_point = current_point * current_point + julia_constant;

        // Once it escapes, this pixel is done.
        if(current_point.magnitude2() > max_magnitude) {
            break;
        }
    }
    return iteration;
}


__global__ void computeJuliaKernel(float *device_julia_set,
        Complex julia_constant, float zoom_scale, int image_width,
        int image_height, int max_iterations, float max_magnitude,
        float horizontal_scale, float vertical_scale) {
    // Map this CUDA thread to exactly one image pixel.
    // x comes from the horizontal block and thread index.
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;

    // y comes from the vertical block and thread index.
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Rounded-up grids create a few extra border threads.
    // Those threads should not touch memory outside the image.
    if(pixel_x >= image_width || pixel_y >= image_height) {
        return;
    }

    // Run the Julia formula and convert the iteration count to a shade.
    int iterations = countJuliaIterations(pixel_x, pixel_y, julia_constant,
            zoom_scale, image_width, image_height, max_iterations,
            max_magnitude, horizontal_scale, vertical_scale);

    // A low value escaped quickly, a high value stayed longer.
    float julia_shade = (float)iterations / max_iterations;

    // Keep the same memory layout as the CPU version.
    device_julia_set[pixel_x * image_height + pixel_y] = julia_shade;
}
}


void julia_kernel(float *host_julia_set, Complex julia_constant,
        float zoom_scale, int image_width, int image_height,
        int max_iterations, float max_magnitude, float horizontal_scale,
        float vertical_scale) {
    // Pick a simple 2D default if the user skipped -b.
    // 16x16 gives 256 threads per block, which is a solid baseline.
    if(global_block_x <= 0 || global_block_y <= 0) {
        global_block_x = 16;
        global_block_y = 16;
    }

    // Reserve enough device memory for the whole image.
    float *device_julia_set = nullptr;

    // Use size_t so very large images do not overflow an int.
    size_t number_of_pixels = (size_t)image_width * (size_t)image_height;

    // Each pixel stores one float shade value.
    size_t number_of_bytes = number_of_pixels * sizeof(float);

    // Allocate the output array on the GPU.
    checkCuda(cudaMalloc((void **)&device_julia_set, number_of_bytes),
            "cudaMalloc");

    // Use the requested block size and round the grid up to cover all pixels.
    dim3 threads_per_block(global_block_x, global_block_y);

    // Integer ceil division gives enough blocks for partial edge blocks too.
    dim3 blocks_per_grid(
            (image_width + threads_per_block.x - 1) / threads_per_block.x,
            (image_height + threads_per_block.y - 1) / threads_per_block.y);

    // Launch one thread per pixel.
    computeJuliaKernel<<<blocks_per_grid, threads_per_block>>>(
            device_julia_set, julia_constant, zoom_scale, image_width,
            image_height, max_iterations, max_magnitude, horizontal_scale,
            vertical_scale);

    // Make sure the kernel finished before the host timing stops.
    // First check whether the launch itself was valid.
    checkCuda(cudaGetLastError(), "kernel launch");

    // Then wait for the actual GPU work to complete.
    checkCuda(cudaDeviceSynchronize(), "kernel execution");

    // Copy the finished image back to the CPU buffer.
    checkCuda(cudaMemcpy(host_julia_set, device_julia_set, number_of_bytes,
                cudaMemcpyDeviceToHost), "copy result to host");

    // Free the temporary GPU buffer for this run.
    checkCuda(cudaFree(device_julia_set), "cudaFree");
}
