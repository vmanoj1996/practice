#include <cuda_runtime.h>
#include<iostream>

using namespace std;

__global__ void sum(int* a, int *b, int N)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index<N)
    {
        b[index] += a[index]+20;
    }
    
}

int main()
{
    int *a, *b;
    const int N = 1e6;

    cudaMalloc(&a, N*sizeof(int));
    cudaMalloc(&b, N*sizeof(int));

    int threadsize = 256;
    int blockCount = (threadsize + N - 1)/threadsize;

    sum<<<blockCount, threadsize>>>(a, b, N);

    int *dst = new int[N];
    cudaMemcpy(dst, b, N, cudaMemcpyDeviceToHost);

    std::cout<<dst[10]<<endl;

    cudaFree(a);
    cudaFree(b);
    free(dst);

    return 0;
}