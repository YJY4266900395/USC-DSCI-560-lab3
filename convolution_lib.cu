// convolution_lib.cu - CUDA convolution shared library for Python
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernels (same as before)
float sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

float sobel_y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

float laplacian[9] = {
     0, -1,  0,
    -1,  4, -1,
     0, -1,  0
};

float sharpen[9] = {
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0
};

float box_blur[9] = {
    0.111f, 0.111f, 0.111f,
    0.111f, 0.111f, 0.111f,
    0.111f, 0.111f, 0.111f
};

float gaussian_5x5[25] = {
    1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
    4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
    6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
    4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
    1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f
};

float box_blur_7x7[49];


// CUDA Kernel
__global__ void convolution2D_kernel(unsigned int *input, unsigned int *output, 
                                      int M, float *kernel, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= M || j >= M) return;
    
    int radius = N / 2;
    float sum = 0.0f;
    
    for (int ki = 0; ki < N; ki++) {
        for (int kj = 0; kj < N; kj++) {
            int img_i = i + ki - radius;
            int img_j = j + kj - radius;
            
            if (img_i < 0) img_i = 0;
            if (img_i >= M) img_i = M - 1;
            if (img_j < 0) img_j = 0;
            if (img_j >= M) img_j = M - 1;
            
            sum += input[img_i * M + img_j] * kernel[ki * N + kj];
        }
    }
    
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    output[i * M + j] = (unsigned int)sum;
}


// Exported functions for Python

// General convolution function
extern "C" void gpu_convolution(unsigned int *h_input, unsigned int *h_output,
                                 int M, float *h_kernel, int N) {
    unsigned int *d_input, *d_output;
    float *d_kernel;
    
    size_t img_size = M * M * sizeof(unsigned int);
    size_t kernel_size = N * N * sizeof(float);
    
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel, kernel_size);
    
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((M + 15) / 16, (M + 15) / 16);
    convolution2D_kernel<<<gridDim, blockDim>>>(d_input, d_output, M, d_kernel, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// Convenience functions using predefined kernels
extern "C" void gpu_sobel_x(unsigned int *h_input, unsigned int *h_output, int M) {
    gpu_convolution(h_input, h_output, M, sobel_x, 3);
}

extern "C" void gpu_sobel_y(unsigned int *h_input, unsigned int *h_output, int M) {
    gpu_convolution(h_input, h_output, M, sobel_y, 3);
}

extern "C" void gpu_laplacian(unsigned int *h_input, unsigned int *h_output, int M) {
    gpu_convolution(h_input, h_output, M, laplacian, 3);
}

extern "C" void gpu_sharpen(unsigned int *h_input, unsigned int *h_output, int M) {
    gpu_convolution(h_input, h_output, M, sharpen, 3);
}

extern "C" void gpu_box_blur(unsigned int *h_input, unsigned int *h_output, int M) {
    gpu_convolution(h_input, h_output, M, box_blur, 3);
}

extern "C" void gpu_gaussian(unsigned int *h_input, unsigned int *h_output, int M) {
    gpu_convolution(h_input, h_output, M, gaussian_5x5, 5);
}

extern "C" void gpu_box_blur_7x7(unsigned int *h_input, unsigned int *h_output, int M) {
    // Initialize 7x7 box blur
    for (int i = 0; i < 49; i++) {
        box_blur_7x7[i] = 1.0f / 49.0f;
    }
    gpu_convolution(h_input, h_output, M, box_blur_7x7, 7);
}