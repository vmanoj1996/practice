#include<exception>
#include <stdexcept>
#include <iostream>
int main()
{
    // throw std::runtime_error("Something went wrong!");
    // disabled using -fno-exceptions in compiler opts

    try{
        int a = 120.0f;
        throw std::runtime_error("Something went wrong!");
    }
    catch(...)
    {
        std::cout<<"something went wrong\n";
    }
}