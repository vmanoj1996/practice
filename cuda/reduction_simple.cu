#include <cuda_runtime.h>
#include<iostream>

// inputs are device pointers
__global__ void reduce_sum(const float* input, float* output, int N)
{   
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared__ float sdata[];

    // local index
    int tid = threadIdx.x;

    // load the data into shared memory
    sdata[tid] = (gid<N)? input[gid] : 0.0f;
    __syncthreads();

    // reduction loop
    for(int layer_size = blockDim.x/2; layer_size>0; layer_size/=2)
    {
        if(tid<layer_size)
        {
            sdata[tid] += sdata[tid + layer_size];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
    
}


int main()
{
    float *a, *b;
    const int N = 256;

    cudaMallocManaged(&a, N*sizeof(int));
    cudaMallocManaged(&b, N*sizeof(int));

    for(int i=0; i<N; i++)
    {
        a[i] = 1;
    }

    int threadsize = 256;
    int blockCount = (threadsize + N - 1)/threadsize;

    reduce_sum<<<blockCount, threadsize, threadsize*sizeof(float)>>>(a, b, N);

    cudaDeviceSynchronize();

    std::cout<<b[0]<<std::endl;

    cudaFree(a);
    cudaFree(b);

    return 0;
}