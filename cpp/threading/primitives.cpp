
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <semaphore>
#include <atomic>
#include <barrier>

// g++ -std=c++20 primitives.cpp && ./a.out

int main()
{
    // MUTEX ----------------------------------------
    std::mutex opengl_mutex;
    
    opengl_mutex.lock();
    // do some gl stuff
    opengl_mutex.unlock();


    // automatic unlocking raii
    {
        std::lock_guard guard(opengl_mutex);
    }

    // SEMAPHORES -----------------------------------
    std::counting_semaphore<5> readerInUse(2);

    readerInUse.acquire(); //reduces
    readerInUse.release(); //increase


    // ATOMICS --------------------------------------
    std::atomic<int> counter = 0;
    counter++; // overloaded to fetch_add(1)


    // READ WRITE LOCK ------------------------------
    // allows multiple reader or one writer at a time. safety basically
    std::shared_mutex rwlock;

    // void readData() 
    // {
    //     std::shared_lock<std::shared_mutex> lock(rwLock);  // Shared access
    //     // Multiple threads can read together
    //     std::cout << sharedData << std::endl;
    // }

    // void writeData(int value) 
    // {
    //     std::unique_lock<std::shared_mutex> lock(rwLock);  // Exclusive access
    //     // Only this thread can write, no readers allowed
    //     sharedData = value;
    // }

    // BARRIER ------------------------------------------------------------
    // similar to __syncthreads(); in cuda
    std::barrier sync_point(5);  // Expects 5 threads


    // void worker(int id) {
    //     // Phase 1: Do some work
    //     std::cout << "Thread " << id << " working...\n";
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100 * id));
        
    //     // Wait for ALL threads to reach this point
    //     sync_point.arrive_and_wait();
        
    //     // Phase 2: Continue only after ALL threads arrive
    //     std::cout << "Thread " << id << " continuing...\n";
    // }


    return 0;
}   