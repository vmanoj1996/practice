#include <iostream>

class Singleton {
private:
    Singleton() = default;
    
public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }
    
    // Delete copy constructor and assignment operator
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    
    void doSomething() {
        std::cout << "Singleton instance called!" << std::endl;
    }
};

int main() {
    Singleton& s1 = Singleton::getInstance();
    Singleton& s2 = Singleton::getInstance();
    
    s1.doSomething();
    std::cout << "Same instance: " << (&s1 == &s2) << std::endl;
    
    return 0;
}