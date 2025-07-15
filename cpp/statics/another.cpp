#include"statics.hpp"

// defining another counter is still fine. statics are file scoped {}
static int counter;
#include <iostream>

int changeSomething()
{
    bla[0] = 100;
    counter = 10000;

    std::cout<<"another static "<<counter<<std::endl;

    return 0;
}
