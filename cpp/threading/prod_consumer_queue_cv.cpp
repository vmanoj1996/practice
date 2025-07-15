#include<iostream>
#include<queue>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<chrono>

std::queue<int> buffer;

std::mutex mtx;
std::condition_variable cv;

bool done = false;

void producer()
{
    using namespace std::chrono_literals;
    for(int i=1; i<=5; i++)
    {
        {
            std::unique_lock lock(mtx);
            buffer.push(i);
            std::cout << "Produced: " << i << std::endl;
        }

        // wake up consumer (how does it know what is consumer)
        cv.notify_one();

        std::this_thread::sleep_for(100ms);
    }
    {
        std::unique_lock lock(mtx);
        done = true;
    }
    // generally the threads wont wake up until the notification is sent. some times spurious wakeups happen
    // notify all wakes every thread.
    cv.notify_one();  // Wake up one waiting thread that uses cv variable
}

void consumer()
{
    while(!done)
    {
        {
            std::unique_lock lock(mtx);

            // block the current thread until the condition inside lambda is met
            // check the predicate while holding the lock.
            // In case false, release the mutex (for other programs to do their work) and goto sleep
            // in case true, keep the lock held and return
            cv.wait(lock, []{ return !buffer.empty() || done; });

            if(!buffer.empty()) 
            {
                int item = buffer.front();
                buffer.pop();
                std::cout << "Consumed: " << item << std::endl;
            }
        }

    }
}


int main()
{

    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();


    return 0;
}