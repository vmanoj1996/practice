template<int N> 
struct Factorial {
    static const int value = N * Factorial<N-1>::value;
};

// Base case - stops the recursion
template<>
struct Factorial<0> {
    static const int value = 1;
};
