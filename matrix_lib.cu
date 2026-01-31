#include <cuda_runtime.h>
#include <cstdio>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiledKernel(
    const float *A, const float *B, float *C, int N)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < tiles; ++m) {
        int aCol = m * TILE_WIDTH + tx;
        int bRow = m * TILE_WIDTH + ty;

        ds_A[ty][tx] = (Row < N && aCol < N) ? A[Row * N + aCol] : 0.0f;
        ds_B[ty][tx] = (bRow < N && Col < N) ? B[bRow * N + Col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// ===== PDF-required interface =====
extern "C"
void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}