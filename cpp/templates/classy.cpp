template<typename T>
struct Handler {
    void process(T data) {
        std::cout << "Generic processing for type\n";
        // Default implementation
    }
};



template<typename T, std::size_t N>
class array {
    T data[N];  // Fixed-size array of type T
    // member functions...
};