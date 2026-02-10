#include <iostream>
#include <random>
#include <cuda_runtime.h>

__global__ void compute_kernel(int *dA, int a) {
    int x = threadIdx.x;  // threadIdx
    int y = blockIdx.x;   // blockIdx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute a*x + y
    dA[idx] = a * x + y;
}

int main() {
    const int n = 16;
    int hA[n];
    int *dA;

    // Generate random integer a
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 20);
    int a = dist(gen);

    // Allocate device memory
    cudaMalloc((void**)&dA, n * sizeof(int));

    // Launch kernel with 2 blocks, 8 threads per block
    compute_kernel<<<2, 8>>>(dA, a);

    // Copy results back to host
    cudaMemcpy(hA, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print all 16 values separated by single space
    for (int i = 0; i < n; i++) {
        std::cout << hA[i];
        if (i < n - 1) std::cout << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(dA);

    return 0;
}
