#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define TILE_WIDTH 16

/**
 * CUDA Tiled Matrix Multiplication Kernel for Attention Projections.
 * 
 * Computes: C = A * B
 * Where A is input sequence activations [M x K]
 *       B is projection weight matrix [K x N] (e.g., Q_proj)
 *       C is projected outputs [M x N]
 * 
 * Optimizes memory throughput by loading sub-blocks (Tiles) of matrices 
 * into high-speed on-chip Shared Memory to minimize global memory thrashing.
 */
__global__ void tiled_attention_projection_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int K, int N) 
{
    // Allocate shared memory for tiles of A and B
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;

    float acc = 0.0f;

    // Loop over the tiles of A and B required to compute the C element
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Collaborative loading of A tile into shared memory
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            s_A[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Collaborative loading of B tile into shared memory
        if (col < N && (t * TILE_WIDTH + ty) < K) {
            s_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Wait for all threads in block to finish loading shared memory
        __syncthreads();

        // Perform dot product multiplication on the loaded tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        // Sync threads before loading next tile
        __syncthreads();
    }

    // Write final accumulated result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Wrapper function to launch the CUDA kernel from host Python/C++ code
extern "C" void launch_tiled_projection(
    const float* h_A, 
    const float* h_B, 
    float* h_C, 
    int M, int K, int N) 
{
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy inputs from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Setup block and grid dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    tiled_attention_projection_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device allocations
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
