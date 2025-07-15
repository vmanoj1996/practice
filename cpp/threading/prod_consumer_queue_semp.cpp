#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <semaphore>
#include <chrono>

const size_t BUFFER_SIZE = 3;

std::queue<int> buffer;
std::mutex buffer_mutex;

// Single semaphore - tracks items available for consumption
std::counting_semaphore<BUFFER_SIZE> items_available{0};

void producer() {
    using namespace std::chrono_literals;
    
    for (int i = 1; i <= 5; ++i) {
        // Wait if buffer is full (manual check)
        while (true) {
            std::unique_lock<std::mutex> lock(buffer_mutex);
            if (buffer.size() < BUFFER_SIZE) {
                buffer.push(i);
                std::cout << "Produced: " << i << " (buffer size: " << buffer.size() << ")\n";
                break;
            }
            lock.unlock();
            std::this_thread::sleep_for(10ms);  // Brief wait before retry
        }
        
        // Signal consumer that item is available
        items_available.release();
        
        std::this_thread::sleep_for(200ms);
    }
}

void consumer() {
    using namespace std::chrono_literals;
    
    for (int i = 0; i < 5; ++i) {
        // Wait for item to be available
        items_available.acquire();
        
        // Remove item from buffer
        int item;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            item = buffer.front();
            buffer.pop();
            std::cout << "Consumed: " << item << " (buffer size: " << buffer.size() << ")\n";
        }
        
        std::this_thread::sleep_for(300ms);
    }
}

int main() {
    std::thread prod(producer);
    std::thread cons(consumer);
    
    prod.join();
    cons.join();
    
    return 0;
}