#include <iostream>

class NoVirtual {
public:
    double data;
};

class WithVirtual {
public:
    double data;
    virtual void func() {}
};

int main() {
    std::cout << "NoVirtual size: " << sizeof(NoVirtual) << " bytes" << std::endl;
    std::cout << "WithVirtual size: " << sizeof(WithVirtual) << " bytes" << std::endl;
    
    return 0;
}