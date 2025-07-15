#include <iostream>
#include <thread>
#include <vector>
#include <numeric>

void sum_range(const std::vector<int>& data, size_t start, size_t end, int& result) {
    result = 0;
    for (size_t i = start; i < end; ++i) {
        result += data[i];
    }
    std::cout << "Thread summed range [" << start << "-" << end << "): " << result << std::endl;
}

int main() {
    // Create test data
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 1);  // Fill with 1,2,3...12
    
    // Results for each thread
    std::vector<int> partial_sums(3);
    std::vector<std::thread> workers;
    
    // FORK: Split work into 3 parallel tasks
    size_t chunk_size = data.size() / 3;
    
    for (int i = 0; i < 3; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == 2) ? data.size() : (i + 1) * chunk_size;  // Handle remainder
        
        // cref is needed deliberately as thread might copy the data internally before starting the threads
        workers.emplace_back(sum_range, std::cref(data), start, end, std::ref(partial_sums[i]));
    }
    
    // JOIN: Wait for all threads to complete
    for (auto& worker : workers) {
        worker.join();
    }
    
    // Combine results
    int total = 0;
    for (int sum : partial_sums) {
        total += sum;
    }
    
    std::cout << "Total sum: " << total << std::endl;
    std::cout << "Expected: " << std::accumulate(data.begin(), data.end(), 0) << std::endl;
    
    return 0;
}