#include<iostream>
using namespace std;

int main()
{
    // RETURN TYPE FUN_NAME = [CAPTURE](args){}
    int age = 10;
    auto adder = [age](int b, int d){
            return age + b +d ;   
        };

    cout<<adder(10, 20)<<endl;

    return 0;
}