#include <cuda_runtime.h>

__global__ void convolve_2d(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    // implementing correlation. 

    // lets handle two dimensions of outputs in each thread separately
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    int n = blockDim.y * blockIdx.y + threadIdx.y;

    int OUTPUT_ROWS = input_rows - kernel_rows + 1;
    int OUTPUT_COLS = input_cols - kernel_cols + 1;

    // check the bounds
    if(m<OUTPUT_ROWS && n<OUTPUT_COLS)
    {
        float sum = 0;
        for(int i=0; i<kernel_rows; i++)
        {
            for(int j=0; j<kernel_cols; j++)
            {
                sum += kernel[i*kernel_cols + j] * input[(i+m)*input_cols + (j+n)];
            }
        }

        output[m*OUTPUT_COLS + n] = sum;
    }
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    const int TC = 16;
    dim3 threads(TC, TC);

    int OUTPUT_ROWS = input_rows - kernel_rows + 1;
    int OUTPUT_COLS = input_cols - kernel_cols + 1;

    dim3 blocks((OUTPUT_ROWS+TC-1)/TC, (OUTPUT_COLS+TC-1)/TC);

    convolve_2d<<<blocks, threads>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);

}