#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(const float* vec1, const float* vec2, float* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        result[idx] = vec1[idx] + vec2[idx];
    }
}

int main() {
    const int size = 10;
    float input1[size], input2[size], output[size];

    float *dev_input1, *dev_input2, *dev_output;
    cudaMalloc(&dev_input1, size * sizeof(float));
    cudaMalloc(&dev_input2, size * sizeof(float));
    cudaMalloc(&dev_output, size * sizeof(float));

    cudaMemcpy(dev_input1, input1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input2, input2, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    addVectors<<<blocksPerGrid, threadsPerBlock>>>(dev_input1, dev_input2, dev_output, size);

    cudaMemcpy(output, dev_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_input1);
    cudaFree(dev_input2);
    cudaFree(dev_output);

    return 0;
}

