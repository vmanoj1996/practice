#include <cuda_runtime.h>

// For high bin counts with less chance of collision
// __global__ void hist_kernel(const int* input, int* histogram, int N, int num_bins)
// {
//     int index = threadIdx.x + blockDim.x * blockIdx.x;
//     if(index<N && input[index]<num_bins)
//     {
//         // are int ops atomic in cuda?
//         // we are accessing shared memory that is shared by several threads with common assignment indices
//         atomicAdd(&histogram[input[index]], 1);
//     }
// }

// For low bin count
__global__ void hist_kernel(const int* input, int* histogram, int N, int num_bins)
{
    extern __shared__ int local_storage[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    // Initialize shared memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        local_storage[i] = 0;
    }
    __syncthreads();

    // Build local histogram - MUST USE ATOMIC!
    if(index < N && input[index] < num_bins) {
        atomicAdd(&local_storage[input[index]], 1);  // FIXED: atomic operation
    }
    __syncthreads();

    // Merge local histogram to global
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (i<num_bins && local_storage[i] > 0) {
            atomicAdd(&histogram[i], local_storage[i]);
        }
    }
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) 
{
    const int TC = 256;

    cudaMemset(histogram, 0, num_bins * sizeof(float));
    int blockcount = (N-1+TC)/TC;
    hist_kernel<<<blockcount, TC, num_bins*sizeof(float)>>>(input, histogram, N, num_bins);

    cudaDeviceSynchronize();
}
