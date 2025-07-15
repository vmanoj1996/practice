class AbstractShape {
public:
    virtual double area() = 0;        // Pure virtual function
    virtual void display() = 0;      // Pure virtual function
    
    // Can have non-pure virtual functions
    virtual double perimeter() {
        return 0.0;
    }
    
    virtual ~AbstractShape() = default;
};

class Triangle : public AbstractShape {
private:
    double base, height, side1, side2, side3;
    
public:
    Triangle(double b, double h, double s1, double s2, double s3) 
        : base(b), height(h), side1(s1), side2(s2), side3(s3) {}
    
    double area() override {          // Must implement
        return 0.5 * base * height;
    }
    
    void display() override {         // Must implement
        std::cout << "Triangle" << std::endl;
    }
    
    double perimeter() override {     // Optional to override
        return side1 + side2 + side3;
    }
};

int main() {
    // AbstractShape shape;          // ❌ ERROR - can't instantiate abstract class
    Triangle tri(10, 8, 6, 8, 10);  // ✅ OK - concrete class
    
    AbstractShape* shape = &tri;     // ✅ OK - pointer to abstract base
    std::cout << shape->area() << std::endl;
}