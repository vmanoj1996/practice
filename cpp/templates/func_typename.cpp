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
