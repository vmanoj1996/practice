
/* VALUE TEMPLATE (NON TYPENAME TEMPLATE): template for size or numbers instead of the more common typename
advantages:
1. Data can get allocated on stack since the size is known at compile time but we dont have to write repeated functions or structs for the same
2. N can additionally be checked during compile time using static assert.
3. const expression will probably be usable for configuring behavior
4. Compiler can optimize code since the sizes are known (loop runrolling for example)
5. Code can be written for AVX or for some stuff with hardware dependant numbers that can be fixed in compile time without changing the code

template<int THREADS_PER_BLOCK>
__global__ void kernel() {
    __shared__ float data[THREADS_PER_BLOCK];  // Size known at compile time
}

kernel<256><<<...>>>();  // 256 is the value, not a type
*/
template<int N>
struct FixedArray 
{ 
    float data[N]; 
};
FixedArray<256> my_array;  // Size known at compile time
