// 2D convolution on CUDA (GPU) implementation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Sobel X - detect vertical edges (N=3)
float sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Sobel Y - detect horizontal edges (N=3)
float sobel_y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

// Laplacian - edge detection (N=3)
float laplacian[9] = {
     0, -1,  0,
    -1,  4, -1,
     0, -1,  0
};

// Sharpen (N=3)
float sharpen[9] = {
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0
};

// Average Blurring (N=3)
float box_blur[9] = {
    0.111f, 0.111f, 0.111f,
    0.111f, 0.111f, 0.111f,
    0.111f, 0.111f, 0.111f
};

// Gaussian blur (N=5)
float gaussian_5x5[25] = {
    1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
    4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
    6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
    4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
    1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f
};

// Box blur (N=7)
float box_blur_7x7[49];    // will be initialized in main


// CUDA Kernel
__global__ void convolution2D_kernel(unsigned int *input, unsigned int *output, 
                                      int M, float *kernel, int N) {
    // Calculate which pixel this thread handles
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    
    // Boundary check
    if (i >= M || j >= M) return;
    
    int radius = N / 2;
    float sum = 0.0f;
    
    // Convolution (same logic as CPU)
    for (int ki = 0; ki < N; ki++) {
        for (int kj = 0; kj < N; kj++) {
            int img_i = i + ki - radius;
            int img_j = j + kj - radius;
            
            // Clamp
            if (img_i < 0) img_i = 0;
            if (img_i >= M) img_i = M - 1;
            if (img_j < 0) img_j = 0;
            if (img_j >= M) img_j = M - 1;
            
            sum += input[img_i * M + img_j] * kernel[ki * N + kj];
        }
    }
    
    // Clamp result
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    output[i * M + j] = (unsigned int)sum;
}


// read .raw image (same as CPU version)
unsigned int* load_raw(const char *filename, int M) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open input file '%s'\n", filename);
        return NULL;
    }
    
    unsigned int *image = (unsigned int *)malloc(M * M * sizeof(unsigned int));
    if (!image) {
        printf("Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }
    
    for (int i = 0; i < M * M; i++) {
        unsigned char pixel;
        if (fread(&pixel, 1, 1, fp) != 1) {
            printf("Error: Failed to read pixel %d\n", i);
            free(image);
            fclose(fp);
            return NULL;
        }
        image[i] = pixel;
    }
    
    fclose(fp);
    return image;
}

// save .raw image (same as CPU version)
int save_raw(const char *filename, unsigned int *image, int M) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot open output file '%s'\n", filename);
        return 0;
    }
    
    for (int i = 0; i < M * M; i++) {
        unsigned char pixel = (unsigned char)(image[i] & 0xFF);
        fwrite(&pixel, 1, 1, fp);
    }
    
    fclose(fp);
    return 1;
}

void print_usage(const char *program_name) {
    printf("Usage: %s -i <input> -o <output> -m <size> -k <kernel>\n\n", program_name);
    printf("Arguments:\n");
    printf("  -i <input>    Input RAW image file (single channel)\n");
    printf("  -o <output>   Output RAW image file\n");
    printf("  -m <size>     Image size (must be square, e.g., 256, 512, 1024)\n");
    printf("  -k <kernel>   Kernel type:\n");
    printf("                  sobel_x   - Sobel X edge detection (3x3)\n");
    printf("                  sobel_y   - Sobel Y edge detection (3x3)\n");
    printf("                  laplacian - Laplacian edge detection (3x3)\n");
    printf("                  sharpen   - Sharpening (3x3)\n");
    printf("                  box       - Box blur (3x3)\n");
    printf("                  gaussian  - Gaussian blur (5x5)\n");
    printf("                  box7      - Box blur (7x7)\n");
    printf("\nExample:\n");
    printf("  %s -i input.raw -o output.raw -m 512 -k sobel_x\n", program_name);
}


int main(int argc, char **argv) {
    char *input_file = NULL;
    char *output_file = NULL;
    int M = 0;
    char *kernel_name = NULL;
    float *h_filter = NULL;  // host (CPU)
    int N = 3;

    // Initialize 7x7 box blur
    for (int i = 0; i < 49; i++) {
        box_blur_7x7[i] = 1.0f / 49.0f;
    }
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!input_file || !output_file || M == 0 || !kernel_name) {
        printf("Error: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Select kernel
    if (strcmp(kernel_name, "sobel_x") == 0) {
        h_filter = sobel_x; N = 3;
    } else if (strcmp(kernel_name, "sobel_y") == 0) {
        h_filter = sobel_y; N = 3;
    } else if (strcmp(kernel_name, "laplacian") == 0) {
        h_filter = laplacian; N = 3;
    } else if (strcmp(kernel_name, "sharpen") == 0) {
        h_filter = sharpen; N = 3;
    } else if (strcmp(kernel_name, "box") == 0) {
        h_filter = box_blur; N = 3;
    } else if (strcmp(kernel_name, "gaussian") == 0) {
        h_filter = gaussian_5x5; N = 5;
    } else if (strcmp(kernel_name, "box7") == 0) {
        h_filter = box_blur_7x7; N = 7;
    } else {
        printf("Error: Unknown kernel '%s'\n\n", kernel_name);
        print_usage(argv[0]);
        return 1;
    }
    
    printf("=== CUDA Convolution ===\n");
    printf("Input:  %s\n", input_file);
    printf("Output: %s\n", output_file);
    printf("Image size: %d x %d\n", M, M);
    printf("Kernel: %s (%d x %d)\n", kernel_name, N, N);
    
    // Load image to CPU memory
    unsigned int *h_input = load_raw(input_file, M);
    if (!h_input) return 1;
    
    unsigned int *h_output = (unsigned int *)malloc(M * M * sizeof(unsigned int));

    // GPU memory
    unsigned int *d_input, *d_output;  // device (GPU)
    float *d_filter;
    
    size_t img_size = M * M * sizeof(unsigned int);
    size_t filter_size = N * N * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_filter, filter_size);
    
    // Copy data from CPU to GPU
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);
    
    // CUDA execution configuration
    // 256 threads per block
    dim3 blockDim(16, 16);
    dim3 gridDim((M + 15) / 16, (M + 15) / 16);
    
    printf("Grid: %d x %d blocks, Block: %d x %d threads\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Timing + Launch kernel
    cudaEventRecord(start);

    convolution2D_kernel<<<gridDim, blockDim>>>(d_input, d_output, M, d_filter, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %f ms (%f seconds)\n", milliseconds, milliseconds / 1000.0);
    
    // Copy result from GPU back to CPU
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    
    // Save output
    if (save_raw(output_file, h_output, M)) {
        printf("Output saved to: %s\n", output_file);
    }
    
    // Free GPU and CPU memo
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    free(h_input);
    free(h_output);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Done!\n");
    return 0;
}
