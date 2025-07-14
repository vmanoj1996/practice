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

__global__ void scale_kernel(float* input, int N, float scale)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_index>=N) return;

    input[global_index] = scale * input[global_index];
}

struct MinOperation
{
    float safe_value = INFINITY;
    __device__ float operator()(float a, float b){return fminf(a, b);}
    __device__ float alloc(float a) {return a;}
};


template<typename Operation>
__global__ void reduce_kernel(const float* input, float* output, int N, Operation op)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_index = threadIdx.x;
    
    // Think about the exit operation

    // declare the shared memory. It is allocated else where
    extern __shared__ float shared_data[];

    // allocate the inputs
    // if(global_index>)
    //  in case of early return, previous garbage may be there in shared memory which will mess up the sum
    shared_data[thread_index] = (global_index < N) ? op.alloc(input[global_index]) : op.safe_value;

    // do not move to the next step before all the threads converge on this stage for this particular block
    __syncthreads();

    // reduce from 256 to 128 to 64 to 32 ... 1
    // assumption the blockDim.x is a power of 2. else we will lose a few numbers by this logic
    for(int layersize = blockDim.x/2; layersize>0; layersize /= 2)
    {   
        if(thread_index<layersize)
        {
            shared_data[thread_index] = op(shared_data[thread_index], shared_data[layersize+thread_index]);
        }
        __syncthreads();
        // initial layer size is 128. finally when the layer size is 1, we would have only two numbers. We dont need the initial count of threads anymore
    }
    
    // when everyone is done. we can assign the output and move on
    // for each block take the data from the leader and assign it to the block index's output
    // we dont have to keep redoing thjis for each thread
    if(thread_index == 0) output[blockIdx.x] = shared_data[0];
}

struct FusedExpMax
{
    float safe_value = INFINITY;
    __device__ float operator()(float a, float b){return fminf(a, b);}
    __device__ float alloc(float a) {return a;}
};


template<typename Operation>
float reduce_operation(float* input, float* output, int N, const Operation& op)
{
    // input is a device pointer of size N. input is modified. please supply a copy because this function will gut it properly lol
    // output is a device pointer of size atleast = blocksPerGrid
    // output could be a larger array. But we need only till blocksPerGrid during each iteration

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // doing recursive reduction as that is more easy to understand and keep things from getting more memory messy
    
    reduce_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(input, output, N, op);
    cudaDeviceSynchronize();

    if(blocksPerGrid == 1)
    {
        // termination condition reached. 1 block only and it gets reduced in one shot

        // since I am swapping input and output buffers. lets send the actual output in as the return value to avoid confusion
        float output_host;
        cudaMemcpy(&output_host, output, 1*sizeof(float), cudaMemcpyDeviceToHost);

        return output_host;
    }
    else
    {
        //  we will still be left with so many blocks that are to be reduced.
        
        // lets call this function again. we will be reducing the new reduced data now
        float *current_input = output;

        // sadly we cannot override the same buffer :P or can we?
        // but if we create new buffers it will be trashing the stack and also create a lot of unnessary storage.
        // we dont need the inputs right? so lemme gut it out
        float *current_output = input;

        return reduce_operation(current_input, current_output, blocksPerGrid, op);
    }
}

struct MaxOperation
{
    float safe_value = -INFINITY;
    __device__ float operator()(float a, float b){return fmaxf(a, b);}
    __device__ float alloc(float a) {return a;}
};

struct SumOperation
{
    float safe_value = 0.0f;
    __device__ float operator()(float a, float b) {return a+b;}
    __device__ float alloc(float a) {return a;}
};


// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

    // lets find the max and rescale. cub style allocation 
    float *temp_input;
    float *temp_output;
    cudaMalloc(&temp_input,  N*sizeof(float));
    cudaMalloc(&temp_output, blocksPerGrid*sizeof(float));

    // alloc input and find the maximum
    cudaMemcpy(temp_input, output, N*sizeof(float), cudaMemcpyDefault);
    float max_value = reduce_operation(temp_input, temp_output, N, MaxOperation());

    // scale operation
    scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, 1.0f/max_value);

    // reduce sum now
    cudaMemcpy(temp_input, output, N*sizeof(float), cudaMemcpyDefault);
    float sum_value = reduce_operation(temp_input, temp_output, N, SumOperation());

    scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, 1.0f/sum_value);

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