#include <cuda_runtime.h>
#include<iostream>

struct DotAddOperation
{
    float safe_value = 0.0f;
    __device__ float operator()(float a, float b) {return a+b;}
    __device__ float alloc(float in1, float in2) {return in1*in2;}
    __device__ float alloc(float in1) {return in1;}
};

template<typename Operation>
__global__ void reduce_kernel(const float* input1, const float* input2, float* output, int N, Operation op)
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
    if(input2 != nullptr) shared_data[thread_index] = (global_index < N) ? op.alloc(input1[global_index], input2[global_index]) : op.safe_value;
    else shared_data[thread_index] = (global_index < N) ? op.alloc(input1[global_index]) : op.safe_value;

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
float reduce_operation(float* input1, const float* input2, float* output, int N, const Operation& op)
{
    // input is a device pointer of size N. input is modified. please supply a copy
    // output is a device pointer of size atleast = blocksPerGrid
    // output could be a larger array. But we need only till blocksPerGrid during each iteration

    float* input = input1;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // doing recursive reduction as that is more easy to understand and keep things from getting more memory messy

    reduce_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(input1, input2, output, N, op);
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

        return reduce_operation(current_input, nullptr, current_output, blocksPerGrid, op);
    }
}

// A, B, result are device pointers
void solve(const float* A, const float* B, float* result, int N)
{
    const int TC = 256;
    int blockcount = (TC-1+N)/TC;
    
    float *temp_input, *temp_output;
    cudaMalloc(&temp_input, N*sizeof(float));
    cudaMalloc(&temp_output, blockcount*sizeof(float));
    cudaMemcpy(temp_input, A, N*sizeof(float), cudaMemcpyDefault);

    float result_host = reduce_operation(temp_input, B, temp_output, N, DotAddOperation());
    cudaMemcpy(result, &result_host, 1*sizeof(float), cudaMemcpyDefault);

    cudaFree(temp_input);
    
    cudaFree(temp_output);

}

int main() {
    int N = 5;
    float h_A[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_B[] = {2.0, 3.0, 4.0, 5.0, 6.0};
    float h_result = 0.0;
    
    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    solve(d_A, d_B, d_result, N);
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Dot product: " << h_result << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    
    return 0;
}