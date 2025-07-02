#include <iostream>

//  g++ classes.cpp -o ./build/a.out && ./build/a.out
class Person
{
    private:
    int age;
    int size;
    int weight;

    std::string name;

    // non owning reference
    Person *parent;

    public:
    // default constructor
    Person(): age(0), size(0), weight(0), parent(nullptr)
    {
        std::cout<<"default constructor called\n";
    }

    // Parameterized constructor
    Person(int age_, int size_, int weight_, std::string name_, Person *parent_): age(age_), size(size_), weight(weight_), parent(parent_), name(name_)
    {
        std::cout<<"Parametrized constructor called\n";
    }

    // copy constructor
    Person(const Person& rhs): age(rhs.age), size(rhs.size), weight(rhs.weight), parent(rhs.parent)
    {
        std::cout<<"Copy constructor called\n";
    }

    // move constructor
    Person(Person && rhs)noexcept: age(rhs.age), size(rhs.size), weight(rhs.weight), name(std::move(rhs.name)), parent(rhs.parent)
    {
        std::cout<<"move constructor called\n";
        rhs.parent = nullptr;
        rhs.age = 0;
        rhs.size = 0;
        rhs.weight = 0;
    }
    
    Person& operator=(const Person& rhs)
    {
        if(this == &rhs) return *this;

        age = rhs.age;
        name = rhs.name;

        std::cout<<"Copy assignment called\n";

        return *this;
    }

    Person& operator=(Person && rhs)
    {
        std::cout<<"move assignment called";
        if(this == &rhs) return *this;
        age = rhs.age;
        name = std::move(rhs.name);

        rhs.age = 0;

        return *this;

    }

    // destructor
    ~Person()
    {
        std::cout<<"destructor called\n";
    }

    void display()
    {
        std::cout<<"\n"<<name<<" "<<size<<" "<<weight<<" \n";
    }

};

int main()
{
    Person a;
    Person b(10, 10, 50, "potato", &a);

    Person c = std::move(b);
    Person d = c;

    d = a;
    d = std::move(a);


    c.display();
    b.display();


}