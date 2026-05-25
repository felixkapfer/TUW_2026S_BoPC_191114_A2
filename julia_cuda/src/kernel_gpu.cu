#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "kernel.hpp"

extern int global_block_x, global_block_y;

namespace {
void checkCuda(cudaError_t result, const char *step) {
    if(result != cudaSuccess) {
        std::cerr << "CUDA error during " << step << ": "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__device__ int checkPointForJuliaSet(int x, int y, Complex c, float scale,
        int res_x, int res_y, int max_iter, float max_mag,
        float x_scale, float y_scale) {
    float scaledX = scale * x_scale * (float)(x - res_x / 2) / (res_x / 2);
    float scaledY = scale * y_scale * (float)(y - res_y / 2) / (res_y / 2);
    Complex z(scaledX, scaledY);

    // Keep iterating while the point still belongs to the candidate set.
    int i = 0;
    for(i = 0; i < max_iter; i++) {
        z = z * z + c;
        if(z.magnitude2() > max_mag) {
            break;
        }
    }
    return i;
}


__global__ void computeJuliaKernel(float *device_julia_set, Complex c,
        float scale, int res_x, int res_y, int max_iter, float max_mag,
        float x_scale, float y_scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Rounded-up grids create extra threads at the image border.
    if(x >= res_x || y >= res_y) {
        return;
    }

    int iter = checkPointForJuliaSet(x, y, c, scale, res_x, res_y,
            max_iter, max_mag, x_scale, y_scale);
    device_julia_set[x * res_y + y] = (float)iter / max_iter;
}
}


void julia_kernel(float *julia_set, Complex c, float scale, int res_x,
        int res_y, int max_iter, float max_mag, float x_scale,
        float y_scale) {
    // Use a basic 2D default block if the user did not pass -b.
    if(global_block_x <= 0 || global_block_y <= 0) {
        global_block_x = 16;
        global_block_y = 16;
    }

    float *device_julia_set = nullptr;
    size_t number_of_pixels = (size_t)res_x * (size_t)res_y;
    size_t number_of_bytes = number_of_pixels * sizeof(float);

    checkCuda(cudaMalloc((void **)&device_julia_set, number_of_bytes),
            "cudaMalloc");

    dim3 block(global_block_x, global_block_y);
    dim3 grid((res_x + block.x - 1) / block.x,
              (res_y + block.y - 1) / block.y);

    computeJuliaKernel<<<grid, block>>>(device_julia_set, c, scale, res_x,
            res_y, max_iter, max_mag, x_scale, y_scale);

    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel execution");
    checkCuda(cudaMemcpy(julia_set, device_julia_set, number_of_bytes,
                cudaMemcpyDeviceToHost), "copy result to host");
    checkCuda(cudaFree(device_julia_set), "cudaFree");
}
