#include <fstream>
#include <iostream>

class FileManager {
public:
    FileManager(const std::string& filename) {
        file.open(filename);           // Acquire resource
        std::cout<<"resource acquired by initialization of class"<<std::endl;

        if (!file.is_open()) 
            throw std::runtime_error("Failed to open file");
    }
    
    ~FileManager() 
    {                   // Destructor automatically called
        std::cout<<"resource deleted by getitng out of scope"<<std::endl;
        if (file.is_open()) {
            file.close();              // Release resource
        }
    }
    
    void write(const std::string& data) {
        file << data;
    }
    
private:
    std::ofstream file;
};

int main() {
    {
        FileManager fm(".test.txt");    // File opened
        fm.write("Hello World");
    }  // Destructor called here - file automatically closed!
    
    // No need to manually close file
    return 0;
}