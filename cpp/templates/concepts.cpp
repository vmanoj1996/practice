// **** Concepts
// Concepts (C++20) are a cleaner way to constrain templates - like "type requirements" that are readable!

#include <concepts>

// Only works with integer types
template<typename T>
requires std::integral<T>
T add(T a, T b) {
    return a + b;
}

// Only works with floating point types  
template<typename T>
requires std::floating_point<T>
T multiply(T a, T b) {
    return a * b;
}

// supports chaining
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Usage:
add(5, 10);        // ✅ Works - int is integral
add(5.5, 2.3);     // ❌ Compile error - double not integral
multiply(5.5, 2.3); // ✅ Works - double is floating point