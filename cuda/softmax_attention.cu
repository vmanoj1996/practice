#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// TODo reduce kernel is not reimplemented for 2d cases properly.

__global__ void exp_kernel(float* output, int N)
{
    // make the exponents and store the intermediates in another variable
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_index<N)
    {
        output[global_index] = expf(output[global_index]);
    }
}



struct MinOperation
{
    float safe_value = INFINITY;
    __device__ float operator()(float a, float b){return fminf(a, b);}
    __device__ float alloc(float a) {return a;}
};


template<typename Operation>
__global__ void reduce_kernel(const float* input, float* output, int M, int N, Operation op)
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
std::vector<float> reduce_operation(float* input, float* output, int M, int N, const Operation& op)
{
    // input is a device pointer of size N. input is modified. please supply a copy because this function will gut it properly lol
    // output is a device pointer of size atleast = blocksPerGrid
    // output could be a larger array. But we need only till blocksPerGrid during each iteration

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // doing recursive reduction as that is more easy to understand and keep things from getting more memory messy
    
    for(int b=0; b<M; b++)
    {
        reduce_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(input, output, M, N, op);
    }
    
    cudaDeviceSynchronize();

    if(blocksPerGrid == 1)
    {
        // termination condition reached. 1 block only and it gets reduced in one shot

        // since I am swapping input and output buffers. lets send the actual output in as the return value to avoid confusion
        std::vector<float> output_host(M);
        cudaMemcpy(output_host.data(), output, M*sizeof(float), cudaMemcpyDeviceToHost);

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

__global__ void minus_kernel(float* input, int N, const float* values, int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < M && col < N) {
        input[row * N + col] -= values[row];
    }
}

__global__ void scale_kernel(float* input, int N, const float* values, int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < M && col < N) {
        input[row * N + col] /= values[row];
    }
}

__global__ void matmul_transposed_scaling_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // output dimension is MxN and collapsed dimension is K

    int m = threadIdx.x + blockIdx.x * blockDim.x; 
    int n = threadIdx.y + blockIdx.y * blockDim.y;

    if(m<M && n<N)
    {
        // valid region in memory
        // reduction is not parallel. hoping that reduction dimension is smaller.
        float sum = 0;
        for(int k=0; k<K; k++)
        {
            float a_mk = A[K*m + k];
            float b_nk = B[K*n + k];
            
            sum += a_mk*b_nk;
        }
        C[N*m + n] = sum/sqrtf(K);
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
void softmax_2d(float* output, int M, int N) 
{
    // Allocate temp buffers
    float *max_values, *sum_values;
    cudaMalloc(&max_values, M * sizeof(float));
    cudaMalloc(&sum_values, M * sizeof(float));
    
    // 1. Find max for each row
    reduce_operation(output, max_values, M, N, MaxOperation());
    
    // 2. Subtract max from each element
    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    minus_kernel<<<blocks, threads>>>(output, N, max_values, M);
    
    // 3. Apply exp
    exp_kernel<<<(M*N + 255) / 256, 256>>>(output, M*N);
    
    // 4. Sum each row
    reduce_operation(output, sum_values, M, N, SumOperation());
    
    // 5. Normalize by sum
    scale_kernel<<<blocks, threads>>>(output, N, sum_values, M);
    
    cudaFree(max_values);
    cudaFree(sum_values);
    cudaDeviceSynchronize();
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
   // 1. Compute Q*K^T (scores)
   /*. 
   Q of size M×d
   key matrix K of size N×d
   value matrix V of size N×d
   */

   const int TC = 16;
   dim3 threadCount(TC, TC);
   dim3 blocksCount((TC+M-1)/TC, (TC+N-1)/TC);

   float *temp_mn;
   cudaMalloc(&temp_mn, M*N*sizeof(float));

   matmul_transposed_scaling_kernel<<<blocksCount, threadCount>>>(Q, K, temp_mn, M, N, d);
   cudaDeviceSynchronize();

   // 2. Apply softmax and scaling row-wise to find the probabilities
    softmax_2d(temp_mn, M, N);



   // 3. Multiply by V


   cudaFree(temp_mn);
   

}

int main() {
   const int M = 3; // Query sequences
   const int N = 4; // Key/Value sequences  
   const int d = 2; // Feature dimension
   
   // Host matrices
   std::vector<float> Q = {1.0f, 2.0f,    // Q[0]: [1, 2]
                          3.0f, 1.0f,    // Q[1]: [3, 1] 
                          2.0f, 3.0f};   // Q[2]: [2, 3]
   
   std::vector<float> K = {1.0f, 1.0f,    // K[0]: [1, 1]
                          2.0f, 0.0f,    // K[1]: [2, 0]
                          0.0f, 2.0f,    // K[2]: [0, 2]
                          1.0f, 2.0f};   // K[3]: [1, 2]
   
   std::vector<float> V = {0.5f, 1.5f,    // V[0]: [0.5, 1.5]
                          1.0f, 2.0f,    // V[1]: [1.0, 2.0]
                          2.0f, 0.5f,    // V[2]: [2.0, 0.5]
                          1.5f, 1.0f};   // V[3]: [1.5, 1.0]
   
   std::vector<float> output(M * d);
   
   // Device pointers
   float *d_Q, *d_K, *d_V, *d_output;
   cudaMalloc(&d_Q, M * d * sizeof(float));
   cudaMalloc(&d_K, N * d * sizeof(float));
   cudaMalloc(&d_V, N * d * sizeof(float));
   cudaMalloc(&d_output, M * d * sizeof(float));
   
   // Copy to device
   cudaMemcpy(d_Q, Q.data(), M * d * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_K, K.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_V, V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
   
   // Call solve function
   solve(d_Q, d_K, d_V, d_output, M, N, d);
   
   // Copy result back
   cudaMemcpy(output.data(), d_output, M * d * sizeof(float), cudaMemcpyDeviceToHost);
   
   // Print results
   std::cout << "Input Q (3x2):\n";
   for (int i = 0; i < M; i++) {
       for (int j = 0; j < d; j++) {
           std::cout << Q[i * d + j] << " ";
       }
       std::cout << "\n";
   }
   
   std::cout << "\nInput K (4x2):\n";
   for (int i = 0; i < N; i++) {
       for (int j = 0; j < d; j++) {
           std::cout << K[i * d + j] << " ";
       }
       std::cout << "\n";
   }
   
   std::cout << "\nInput V (4x2):\n";
   for (int i = 0; i < N; i++) {
       for (int j = 0; j < d; j++) {
           std::cout << V[i * d + j] << " ";
       }
       std::cout << "\n";
   }
   
   std::cout << "\nOutput (3x2):\n";
   for (int i = 0; i < M; i++) {
       for (int j = 0; j < d; j++) {
           std::cout << output[i * d + j] << " ";
       }
       std::cout << "\n";
   }
   
   // Cleanup
   cudaFree(d_Q);
   cudaFree(d_K);
   cudaFree(d_V);
   cudaFree(d_output);
   
   return 0;
}