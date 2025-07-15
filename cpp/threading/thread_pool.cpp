// Use a pool of N threads to solve some task of size bla

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

std::queue<int> task_queue;
std::mutex mtx;
std::condition_variable cv;

void worker(int worker_id) {
    while (true) {
        int task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [] { return !task_queue.empty(); });
            
            task = task_queue.front();
            task_queue.pop();
        }
        
        // Do the work
        std::cout << "Worker " << worker_id << " processing task " << task << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // without sleep, it can starve other workers
    }
}

int main() {
    // Create 3 worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < 3; ++i) {
        workers.emplace_back(worker, i);
    }
    
    // Submit 10 tasks
    for (int task = 1; task <= 10; ++task) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            task_queue.push(task);
        }
        cv.notify_one();  // Wake up a worker
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // Let workers finish current tasks
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    return 0;  // Note: workers keep running (infinite loop)
}