#include <stdio.h>
#include <stdlib.h>

// Error checking macro for CUDA API calls
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

// Kernel 1: Row-wise, each thread handles one row (less efficient)
__global__ void MatrixAdd_C(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}

// Kernel 2: 2D thread grid, each thread handles one element (most efficient)
__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) { // Fixed condition (original had incorrect &&)
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

// Kernel 3: Column-wise, each thread handles one column (less efficient)
__global__ void MatrixAdd_D(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < N) {
        for (int i = 0; i < N; i++) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}

int main() {
    // Changed variables
    const int N = 16; // Larger matrix size (16x16 instead of 10x10)
    float *A, *B, *C;

    // Allocate host memory
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    // Initialize matrices with different values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)(i + j);     // A[i,j] = i + j (e.g., 0, 1, 2, ...)
            B[i * N + j] = (float)(i * j);     // B[i,j] = i * j (e.g., 0, 0, 0, ...)
            C[i * N + j] = 0.0f;               // Initialize C to 0
        }
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, N * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure grid and block sizes (optimized for larger N)
    dim3 dimBlock(16, 16); // 16x16 threads per block (256 threads)
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    MatrixAdd_B<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print matrices
    printf("Matrix C (Result):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }
    printf("\nMatrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(A);
    free(B);
    free(C);

    return 0;
}
