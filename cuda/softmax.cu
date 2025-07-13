#include <cuda_runtime.h>
#include <iostream>

__global__ void exp_kernel(const float* input, float* output, int N)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_index<N)
    {
        output[global_index] = expf(input[global_index]);
    }
}

__global__ void reduce_max_kernel(const float* input, float* output, int N)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_index>N) return;

    // simple reduce kernel that makes use of shared memory between the threads (registers are even faster)



    // Find the max number
    
    // Scale all the numbers with that max

    // Perform reduce sum on that thing

    // do the division operation

    // set the outputs
}

void reduce_max(const float* input)
{

}

__global__ void scale_kernel(float* input, int N, float scale)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_index>N) return;

    input[global_index] = scale * input[global_index];
}

__global__ void reduce_kernel(const float* input, float* output, int N)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_index = threadIdx.x;
    
    // Think about the exit operation
    if(global_index>=N) return;

    // declare the shared memory. It is allocated else where
    __shared__ float shared_data[blockDim.x];

    // allocate the inputs
    // if(global_index>)
    shared_data[threadIdx.x] = input[global_index];

    // do not move to the next step before all the threads converge on this stage for this particular block
    __syncthreads();

    // reduce from 256 to 128 to 64 to 32 ... 1
    // assumption the blockDim.x is a power of 2. else we will lose a few numbers by this logic
    for(int layersize = blockDim.x/2; layersize>1; layersize /= 2)
    {   
        if(thread_index<layersize)
        {
            shared_data[thread_index] += shared_data[layersize+thread_index];
            __syncthreads();
        }
        // initial layer size is 128. finally when the layer size is 1, we would have only two numbers. We dont need the initial count of threads anymore
    }
    
    // when everyone is done. we can assign the output and move on
    *output = shared_data[0];
}

float reduce_operation(const float* input)
{
    // input is a device pointer
    float host_result;
    

}



// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);


    cudaDeviceSynchronize();
}


int main() {
    const int N = 3;
    float host_input[N] = {1.0f, 2.0f, 3.0f};
    float host_output[N];
    
    float *device_input, *device_output, *device_temp;
    
    // Allocate device memory
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_temp, N * sizeof(float));
    cudaMalloc(&device_output, N * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Call solve function
    solve(device_input, device_output, N);
    
    // Copy result back to host
    cudaMemcpy(host_output, device_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    for(int i = 0; i < N; i++) {
        std::cout << "Output[" << i << "] = " << host_output[i] << std::endl;
    }
    
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    
    return 0;
}