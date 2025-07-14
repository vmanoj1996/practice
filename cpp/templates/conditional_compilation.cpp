#include <type_traits>

// Only enabled for floating point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
sqrt_func(T x) {
    return std::sqrt(x);  // Use optimized floating point sqrt
}

// Only enabled for integer types  
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
sqrt_func(T x) {
    return static_cast<T>(std::sqrt(static_cast<double>(x)));  // Convert to double first
}

// Usage:
sqrt_func(4.5f);   // Calls floating point version
sqrt_func(16);     // Calls integer version  
sqrt_func("hi");   // Compile error - no matching function



