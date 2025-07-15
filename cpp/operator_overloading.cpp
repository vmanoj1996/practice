#include <iostream>
#include<thread>

class Number {
    int value;
public:
    Number(int v) : value(v) {}
    
    Number operator+(const Number& rhs) {
        return Number(rhs.value);
    }
    
    int getValue() const { return value; }
};


int main()
{
    Number a(1), b(2);

    Number c = a+b;

    std::cout<<c.getValue()<<std::endl;

    return 0;
}
