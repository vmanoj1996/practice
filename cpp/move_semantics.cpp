

// already practised in classes.cpp

#include <iostream>


int main()
{
    int a = 1000;
    // int b = (int&&) a;
    int b = std::move(a);

    // for primijtive types this does not actually move. just copies. for custom stuff, a default move constructor or move sassignment operator is provided
    std::cout<<"a:"<<a<<" b:"<<b<<std::endl;

    return 0;
}