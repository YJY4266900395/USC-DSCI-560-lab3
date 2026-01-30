// 2D convolution on CPU implementation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

// TODO: Implement the convolution function here
void convolution2D(unsigned int *input, unsigned int *output, int M, float *kernel, int N) {
    // M = image size M*M
    // N = kernel size N*N
    // TODO: Convolution implementation
    int radius = N / 2;

    // loop all output i,j
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            // loop all i,j in kernel
            for (int ki = 0; ki < N; ki++) {
                for (int kj = 0; kj < N; kj++) {
                    // corresponding input image i,j
                    int img_i = i + ki - radius;
                    int img_j = j + kj - radius;

                    // clamp the margin
                    if (img_i < 0) img_i = 0;
                    if (img_i >= M) img_i = M - 1;
                    if (img_j < 0) img_j = 0;
                    if (img_j >= M) img_j = M - 1;

                    // add img * kernel to sum
                    sum += input[img_i * M + img_j] * kernel[ki * N + kj];
                }
            }

            // clamp result to 0-255
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[i * M + j] = (unsigned int) sum;
        }
    }
}


int main(int argc, char **argv) {
    // default arguments
    int M = 512;  // image size
    int N = 3;    // kernel size
    float *filter = sobel_x;
    
    // argument for image size
    if (argc > 1) {
        M = atoi(argv[1]);
    }
    
    printf("=== CPU Convolution ===\n");
    printf("Image size: %d x %d\n", M, M);
    printf("Convolution kernel size: %d x %d\n", N, N);
    
    // assign space
    unsigned int *input = (unsigned int *)malloc(M * M * sizeof(unsigned int));
    unsigned int *output = (unsigned int *)malloc(M * M * sizeof(unsigned int));
    
    if (!input || !output) {
        printf("Error: memory allocation failed\n");
        return 1;
    }

    // simple test data
    for (int i = 0; i < M * M; i++) {
        input[i] = rand() % 256;
    }
    
    // convolution w/ timing
    clock_t start = clock();
    convolution2D(input, output, M, sobel_x, 3);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", elapsed);

    
    // release space
    free(input);
    free(output);
    
    printf("Done!\n");
    return 0;
}