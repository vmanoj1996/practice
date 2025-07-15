#include <iostream>
#include <memory>

std::shared_ptr<int> shared1; // this would get shadowed if a local exists

int main()
{
    int b = 1;

    int *c = &b;

    // illegal. cannot own it uniquely. cannot convert raw pointer to unique pointer directly.
    // std::unique_ptr<int> bla = &b;

    // call the make unique factory method
    std::unique_ptr<int> bla = std::make_unique<int>(2158);

    // try copy constructor (copy constructor is deleted)
    // std::unique_ptr<int> ble = bla;

    // try move constructor
    std::unique_ptr<int> ble = std::move(bla);    

    std::cout<<" pointer stores="<<*ble<<std::endl;


    // shared pointer
    std::shared_ptr<int> shared1= std::make_shared<int>(200);
    {
        auto shared2 = shared1;
        std::cout<<"count: "<<shared1.use_count()<<std::endl;
    }
    std::cout<<"count: "<<shared1.use_count()<<std::endl;

    std::cout<<"count: "<<::shared1.use_count()<<std::endl;


    // new operator
    std::unique_ptr<int> ptr(new int(42));

    std::unique_ptr<int[]> ptrs(new int[100]{45});

    auto ptr1 = std::make_unique<int[]>(1000);

    std::cout<<ptr1[0]<<std::endl;


    return 0;

}
