#include <vector>
#include <iostream>

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K)
{
    // reduction dimension is n
    // m, n, k
    int k = threadIdx.x + blockIdx.x * blockDim.x; 
    int m = threadIdx.y + blockIdx.y * blockDim.y;

    if(m<M && k<K)
    {
        // valid region in memory
        // reduction is not parallel. hoping that reduction dimension is smaller.
        float sum = 0;
        for(int n=0; n<N; n++)
        {
            float a_kn = A[N*m + n];
            float b_nk = B[K*n + k];
            
            sum += a_kn*b_nk;
        }
        C[K*m + k] = sum;
    }
}


int main() {
   const int M = 4, N = 3, K = 2;
   
   // Host matrices
   std::vector<float> A = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9,
                          10, 11, 12};
   
   std::vector<float> B = {1, 2,
                          3, 4,
                          5, 6};
   
   std::vector<float> C(M * K);
   
   // Device pointers
   float *d_A, *d_B, *d_C;
   cudaMalloc(&d_A, M * N * sizeof(float));
   cudaMalloc(&d_B, N * K * sizeof(float));
   cudaMalloc(&d_C, M * K * sizeof(float));
   
   // Copy to device
   cudaMemcpy(d_A, A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
   
   // Launch kernel
   dim3 blockDim(16, 16);
   dim3 gridDim((K + 15) / 16, (M + 15) / 16);
   matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
   
   // Copy result back
   cudaMemcpy(C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
   
   // Print result
   std::cout << "Result (4x2):\n";
   for (int i = 0; i < M; i++) {
       for (int j = 0; j < K; j++) {
           std::cout << C[i * K + j] << " ";
       }
       std::cout << "\n";
   }
   
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   
   return 0;
}