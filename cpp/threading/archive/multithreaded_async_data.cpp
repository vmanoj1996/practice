#include <iostream>
#include <thread>
#include <vector>
#include <array>
#include <atomic>
#include <chrono>
#include <numeric> // Still needed for std::fill, if not removing all uses of it

using namespace std;
using namespace std::chrono_literals;

std::atomic<bool> terminate_flag = false;

std::atomic<int> active_buffer_index = 0;

static std::array<std::array<uint8_t, 5>, 5> display_buffers[2];

void bg_render_task()
{
    while (!terminate_flag)
    {
        int current_read_buffer_idx = active_buffer_index.load(std::memory_order_acquire);

        // Simulate reading each element
        for (const auto& row : display_buffers[current_read_buffer_idx])
        {
            for (uint8_t val : row)
            {
                // In a real scenario, 'val' would be used for rendering
            }
        }

        std::this_thread::sleep_for(200ms);
    }
}

int main()
{
    // Initialize buffers manually without std::iota for clearer learning
    uint8_t current_init_val = 0;
    for (int i = 0; i < 2; ++i) { // Loop through each of the two display buffers
        for (size_t row_idx = 0; row_idx < display_buffers[i].size(); ++row_idx) { // Loop through rows
            for (size_t col_idx = 0; col_idx < display_buffers[i][row_idx].size(); ++col_idx) { // Loop through columns
                display_buffers[i][row_idx][col_idx] = current_init_val++;
            }
        }
    }

    auto background_render_thread = std::thread(bg_render_task);

    int update_count = 0;
    while (update_count < 5)
    {
        int current_active_idx = active_buffer_index.load(std::memory_order_relaxed);
        int inactive_buffer_idx = (current_active_idx == 0) ? 1 : 0;

        uint8_t fill_value = (update_count + 1) * 10;
        // Manually fill the inactive buffer without std::fill for clearer learning
        for (size_t row_idx = 0; row_idx < display_buffers[inactive_buffer_idx].size(); ++row_idx) {
            for (size_t col_idx = 0; col_idx < display_buffers[inactive_buffer_idx][row_idx].size(); ++col_idx) {
                display_buffers[inactive_buffer_idx][row_idx][col_idx] = fill_value;
            }
            // Increment fill_value per row, if desired, otherwise all elements in the buffer will be the same
            // For example, fill_value++;
        }


        active_buffer_index.store(inactive_buffer_idx, std::memory_order_release);

        std::this_thread::sleep_for(400ms);
        update_count++;
    }

    terminate_flag = true;

    background_render_thread.join();

    return 0;
}