#include <iostream>
#include<thread>
#include <vector>
#include<array>
#include<atomic>

using namespace std;

// concepts:
// design patterns. 
// async, fork join, producer-consumer, superloop

// mutex, semaphore

std::atomic<bool> terminate_flag = false;
std::atomic<bool> activeBuffer = false;

static std::array<std::array<uint8_t, 100>, 100> displaybuffer[2];

void bg()
{
    while(!terminate_flag)
    {
        cout<<"inside the task"<<endl;
        std::this_thread::sleep_for(100ms);
    }
    std::cout<<"terminate flaf called"<<endl;
}

int main()
{
    {
        //background render thead that runs asynchronous based on the latest rendering data
        auto background_render_thread = std::thread(bg);

        std::this_thread::sleep_for(3s);
        terminate_flag = true;
        background_render_thread.join();
    }

    return 0;
}
