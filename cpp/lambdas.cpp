#include<iostream>
#include <iomanip>
using namespace std;

int main()
{
    // RETURN TYPE FUN_NAME = [CAPTURE](args){}
    int age = 10;
    auto adder = [age](int b, int d){
            return age + b +d ;   
        };


    float beta = -1e-5;
    auto suppy = [beta](float a, float b)
    {
        return a-b+beta;
    };

    cout << std::fixed << std::setprecision(10);
    cout<<adder(10, 20)<<" "<<suppy(1.0f, 2.0f)<<endl;

    return 0;
}