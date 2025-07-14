#include <cuda_runtime.h>
#include <iostream>

// Clang build
//          nvcc -arch=sm_89 -ccbin clang++ reduction_generic.cu -o a.out

// gcc nvcc build
//         nvcc -arch=sm_89 reduction_generic.cu

// tools to check the mangled symbols
//           nm -C a.out | grep reduce_kernel
//           cuobjdump -symbols a.out

struct SumOperation
{
    float safe_value = 0.0f;
    __device__ float operator()(float a, float b) {return a+b;}
    __device__ float alloc(float a) {return a;}
};

struct MaxOperation
{
    float safe_value = -INFINITY;
    __device__ float operator()(float a, float b){return fmaxf(a, b);}
    __device__ float alloc(float a) {return a;}
};

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

template<typename Operation>
float reduce_operation(float* input, float* output, int N, const Operation& op)
{
    // input is a device pointer of size N. input is modified. please supply a copy
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

int main() {
    const int N = 1e6;

    float host_input[N];
    
    // Initialize inputs as 1, 2, 3, 4, ..., N
    for(int i = 0; i < N; i++) {
        host_input[i] = i + 1;
    }
    
    float *device_input;
    
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout<<"execution started\n";
    int threadsPerBlock = 256;  
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float* output_buffer;
    cudaMalloc(&output_buffer, blocksPerGrid*sizeof(float));

    float sum = reduce_operation(device_input, output_buffer, N, SumOperation()); cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    float min = reduce_operation(device_input, output_buffer, N, MinOperation()); cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    float max = reduce_operation(device_input, output_buffer, N, MaxOperation());
    
    std::cout << "Sum = " << sum<<" Min = " << min<<" Max = " << max << std::endl;
    
    float N_float = static_cast<float>(N);
    std::cout << "Expected sum= " << N_float*(N_float+1)/2 << std::endl;

    cudaFree(device_input);
    cudaFree(output_buffer);
    
    return 0;
}