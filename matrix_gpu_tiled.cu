#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // number of tiles to iterate over
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m) {
        int tiledColA = m * TILE_WIDTH + tx; // column in A
        int tiledRowB = m * TILE_WIDTH + ty; // row in B

        // Load tile from A into shared memory
        if (Row < N && tiledColA < N)
            ds_A[ty][tx] = A[Row * N + tiledColA];
        else
            ds_A[ty][tx] = 0.0f;

        // Load tile from B into shared memory
        if (Col < N && tiledRowB < N)
            ds_B[ty][tx] = B[tiledRowB * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(0);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void**)&d_A, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc((void**)&d_B, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc((void**)&d_C, bytes), "cudaMalloc C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "EventCreate start");
    checkCuda(cudaEventCreate(&stop), "EventCreate stop");

    checkCuda(cudaEventRecord(start), "EventRecord start");
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaEventRecord(stop), "EventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "EventSync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "ElapsedTime");

    // optional: copy back (not included in kernel timing above)
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    printf("Tiled CUDA kernel time (N=%d): %.3f ms\n", N, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
