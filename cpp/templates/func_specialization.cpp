/* Generic handler and more specialized float handler

template<> means fully specialized as indicated later on

*/ 
template<typename T>
struct Handler {
    void process(T data) {
        std::cout << "Generic processing for type\n";
        // Default implementation
    }
};

// Specialized version for float
template<>
struct Handler<float> {
    void process(float data) {
        std::cout << "Optimized float processing with SIMD\n";
        // Use SSE/AVX instructions for floats
    }
};

// Specialized version for int
template<>
struct Handler<int> {
    void process(int data) {
        std::cout << "Integer-specific bit manipulation\n";
        // Use bit tricks for integers
    }
};

// Usage:
Handler<double> h1;  // Uses generic version
Handler<float> h2;   // Uses float specialization
Handler<int> h3;     // Uses int specialization

h1.process(3.14);    // "Generic processing"
h2.process(3.14f);   // "Optimized float processing"
h3.process(42);      // "Integer-specific bit manipulation"