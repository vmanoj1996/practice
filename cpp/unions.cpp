
#include <iostream>
#include <variant>

union MyUnion
{
    int i;
    float j;
    char c;
};

// MODERN C++ Variant
std::variant<int, float, char> data2;


int main()
{
    MyUnion d;
    d.i = 61;


    data2 = 42;

    std::cout<<d.j<<"\n";
    std::cout<<std::get<int>(data2)<<std::endl; //correct match
    std::cout<<std::get<float>(data2)<<std::endl; // incorrect stored variant type -> throws

    return 0;
}


