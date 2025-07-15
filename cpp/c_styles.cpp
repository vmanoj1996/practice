// #define: Text replacement, no type checking
// typedef: Type aliases (use using in modern C++)
// Conditional compilation: Still useful for platform-specific code
// Macros: Powerful but dangerous - prefer constexpr functions

#define PI 3.14159
#define MAX_SIZE 1000

// function like macro
#define SQUARE(x) ((x) * (x))

typedef int* IntPtr;
typedef unsigned long long ULL;
typedef struct {
    int x, y;
} Point;


#define DEBUG 1

#ifdef DEBUG
    #define LOG(x) std::cout << x << std::endl
#else
    #define LOG(x)  // Empty - no logging in release
#endif


// #pragma once  


#pragma pack(1)                 // Structure packing
struct PackedStruct {
    char c;
    int i;
};
#pragma pack()