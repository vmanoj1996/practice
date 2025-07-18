#include <cuda_runtime.h>
#include<iostream>
#include <bit>
#include <vector>
#include <limits>

// nvcc -std=c++20 -arch=sm_89 sort_bitonic.cu && ./a.out


void print_device_array(float* device_data, int N) {
    float* host_data = new float[N];
    cudaMemcpy(host_data, device_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
    
    delete[] host_data;
}


__global__ void bitonic_kernel(float* data_app_device, const int N_appended, const int level, const int slab_size, const int distance)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if(gid<N_appended)
    {
        int pair = gid^distance; // get gid+distance or gid-distance automagically within the current slab
        if(gid<pair)
        {
            // no two threads will need to access the same pair data
            int slab_index = gid/slab_size;
            bool shouldIncrease = (slab_index%2 == 0); //even means increase

            if(data_app_device[gid]>data_app_device[pair] == shouldIncrease)
            {
                float tmp = data_app_device[gid];
                data_app_device[gid] = data_app_device[pair];
                data_app_device[pair] = tmp;
            }
        }
    }

}

// data is device pointer
void solve(float* data, int N) 
{
    // single bit check
    // std::cout<<std::has_single_bit(static_cast<unsigned>(L))<<std::endl;

    // // check xor
    // int index = 14;
    // int distance = 4;
    // std::cout<<(index^distance)<<std::endl;

    // Pseudo Code: -------------------------------------------------------------
    // make the array exactly 2^L=N by appending inf at the end

    // make gpu array 

    // for levels 1, 2, 3, 4, .l,.. L
    //      for distance l/2, l/4, l/8 ... 1
    //            run the bitonic kernel for this level and this distance
    //            Sync the device

    // Copy the array back to CPU

    // Delete the temporaries

    // --------------------------------------------------------------------------

    std::vector<float> data_app_host(N);
    cudaMemcpy(data_app_host.data(), data, N*sizeof(float), cudaMemcpyDefault);
    int N_appended = N;

    auto has_single_bit = [](int N) {
        return N > 0 && (N & (N - 1)) == 0;
    };

    if(!has_single_bit(static_cast<unsigned>(N)))
    {
        N_appended = pow(2, ceil(log2(N)));
        data_app_host.resize(N_appended);

        std::fill(data_app_host.data()+N, data_app_host.data() + N_appended, std::numeric_limits<float>::max());
    }

    // for(auto elem:data_app_host)
    // {
    //     std::cout<<elem<<" ";
    // }
    // std::cout<<"\n";

    float *data_app_device;
    cudaMalloc(&data_app_device, N_appended*sizeof(float));
    cudaMemcpy(data_app_device, data_app_host.data(), N_appended*sizeof(float), cudaMemcpyDefault);

    // lets do the levels now
    for(int slab_size = 2; slab_size<=N_appended; slab_size*=2)
    {
        // std::cout<<"slab size "<<slab_size;
        for(int distance = slab_size/2; distance>=1; distance /=2)
        {
            // std::cout<<"    distance "<<distance<<" ";
            const int TC = 256;
            int blocks = (TC-1+N_appended)/TC;

            int level = log2(slab_size);
            bitonic_kernel<<<blocks, TC>>>(data_app_device, N_appended, level, slab_size, distance);
            cudaDeviceSynchronize();

            // print_device_array(data_app_device, N_appended);
        }

        // std::cout<<std::endl;
    }

    cudaMemcpy(data, data_app_device, N*sizeof(float), cudaMemcpyDefault);

}



int main() {
    // Test case 1
    std::vector<float> data1_vec = {5.0, -100, 0.5, 2.0, 8.0, 1.0, 9.0, 4.0, 21, 12, 23454, 32, 12, 12, 3425,3, 2, 1212, 432, 23};
    int N1 = data1_vec.size();

    std::cout << "Input : ";
    for (int i = 0; i < N1; i++) {
        std::cout << data1_vec[i] << " ";
    }
    std::cout << std::endl;
    
    float* data1 = data1_vec.data();
    
    float* d_data1;
    cudaMalloc(&d_data1, N1 * sizeof(float));
    cudaMemcpy(d_data1, data1, N1 * sizeof(float), cudaMemcpyHostToDevice);
    
    solve(d_data1, N1);
    
    cudaMemcpy(data1, d_data1, N1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data1);
    
    std::cout << "Result: ";
    for (int i = 0; i < N1; i++) {
        std::cout << data1[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

