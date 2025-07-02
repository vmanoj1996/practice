#include<iostream>

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

    using namespace std;
    int *a = new int;

    *a = 1;
    a[0] = 2;
    cout<<*a<<endl;
    delete a;
    a = nullptr;


    int *b = new int[100];
    b[2] = 100;
    cout<<b[2]<<endl;
    delete[] b;
    b = nullptr;


    Person *persons = new Person[10];
    delete[] persons;
    persons = nullptr;


    return 0;
}