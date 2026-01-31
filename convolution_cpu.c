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


void convolution2D(unsigned int *input, unsigned int *output, int M, float *kernel, int N) {
    // M = image size M*M
    // N = kernel size N*N
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

// read .raw image
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
    
    // read every pixel
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

// save .raw image after convolution
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
    // default arguments
    char *input_file = NULL;
    char *output_file = NULL;
    int M = 0;
    char *kernel_name = NULL;
    float *filter = NULL;
    int N = 3;

    // initialize 7x7 box blur
    for (int i = 0; i < 49; i++) {
        box_blur_7x7[i] = 1.0f / 49.0f;
    }
    
    // arguments from command line
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

    // check necessary arguments
    if (!input_file || !output_file || M == 0 || !kernel_name) {
        printf("Error: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // select kernel
    if (strcmp(kernel_name, "sobel_x") == 0) {
        filter = sobel_x; N = 3;
    } else if (strcmp(kernel_name, "sobel_y") == 0) {
        filter = sobel_y; N = 3;
    } else if (strcmp(kernel_name, "laplacian") == 0) {
        filter = laplacian; N = 3;
    } else if (strcmp(kernel_name, "sharpen") == 0) {
        filter = sharpen; N = 3;
    } else if (strcmp(kernel_name, "box") == 0) {
        filter = box_blur; N = 3;
    } else if (strcmp(kernel_name, "gaussian") == 0) {
        filter = gaussian_5x5; N = 5;
    } else if (strcmp(kernel_name, "box7") == 0) {
        filter = box_blur_7x7; N = 7;
    } else {
        printf("Error: Unknown kernel '%s'\n\n", kernel_name);
        print_usage(argv[0]);
        return 1;
    }
    
    printf("=== CPU Convolution ===\n");
    printf("Input:  %s\n", input_file);
    printf("Output: %s\n", output_file);
    printf("Image size: %d x %d\n", M, M);
    printf("Kernel: %s (%d x %d)\n", kernel_name, N, N);
    
    // read input image
    unsigned int *input = load_raw(input_file, M);
    if (!input) {
        return 1;
    }

    // assign output space
    unsigned int *output = (unsigned int *)malloc(M * M * sizeof(unsigned int));
    if (!output) {
        printf("Error: Memory allocation failed\n");
        free(input);
        return 1;
    }

    
    // convolution w/ timing
    printf("Processing...\n");
    clock_t start = clock();
    convolution2D(input, output, M, filter, N);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", elapsed);

    // save output image
    if (save_raw(output_file, output, M)) {
        printf("Output saved to: %s\n", output_file);
    }

    // release space
    free(input);
    free(output);
    
    printf("Done!\n");
    return 0;
}