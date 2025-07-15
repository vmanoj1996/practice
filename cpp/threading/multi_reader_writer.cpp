#include <iostream>
#include <thread>
#include <shared_mutex>
#include <vector>
#include <chrono>
#include <mutex>

std::shared_mutex rwLock;
int sharedData = 0;

void reader(int id) {
    using namespace std::chrono_literals;
    
    for (int i = 0; i < 3; ++i) {
        // Acquire shared lock for reading
        std::shared_lock lock(rwLock);
        
        std::cout << "Reader " << id << " reads: " << sharedData << std::endl;
        std::this_thread::sleep_for(200ms);  // Simulate read work. lock is held
        
        // Lock automatically released when scope ends
    }
}

void writer(int id) {
    using namespace std::chrono_literals;
    
    for (int i = 0; i < 2; ++i) {
        // Acquire exclusive lock for writing
        std::unique_lock lock(rwLock);
        
        sharedData += 10;
        std::cout << "Writer " << id << " writes: " << sharedData << std::endl;
        std::this_thread::sleep_for(300ms);  // Simulate write work
        
        // Lock automatically released when scope ends
    }
}

int main() {
    std::vector<std::thread> threads;
    
    // Start 3 reader threads
    for (int i = 1; i <= 3; ++i) {
        threads.emplace_back(reader, i);
    }
    
    // Start 2 writer threads  
    for (int i = 1; i <= 2; ++i) {
        threads.emplace_back(writer, i);
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Final value: " << sharedData << std::endl;
    
    return 0;
}