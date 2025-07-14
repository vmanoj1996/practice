
//  Variadic templates with unknown count of args and unknown count of types

template<typename... Args> //unknown number of types
void print(Args... args) // unknown number of args
{
    ((std::cout << args << " "), ...);  // Expands to: cout << arg1 << " ", cout << arg2 << " ", ...
}

print(42);                    // Args = {int}
print(42, 3.14f);            // Args = {int, float}  
print(42, 3.14f, "hello");   // Args = {int, float, const char*}



//  Variadic templates with 1 unknown argtype and unknown count of args

template<typename T>
void print_same_type(T... args) {
    ((std::cout << args << " "), ...);
}

print_same_type(1, 2, 3, 4);        // All ints
print_same_type(1.1, 2.2, 3.3);     // All floats

//  Variadic templates with 1 type and 1 arg but count of array unknown

template<int N>
void print_n_ints(int (&arr)[N]) {
    for(int i = 0; i < N; ++i) {
        std::cout << arr[i] << " ";
    }
}

int data[] = {1, 2, 3};
print_n_ints(data);  // N deduced as 3


/*
initializer list (type is always known at compile time. values and length runtime)
std::initializer_list is a lightweight container that holds a list of values of the same type:
*/ 

void print_ints(std::initializer_list<int> values) {
    for(auto v : values) {
        std::cout << v << " ";
    }
}

// Usage:
print_ints({1, 2, 3, 4, 5});        // Curly braces create initializer_list
print_ints({10, 20});               // Variable number of args
print_ints({});                     // Empty list