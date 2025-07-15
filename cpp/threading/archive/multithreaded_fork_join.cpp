#include <iostream>
#include<thread>
#include <vector>
#include<array>

using namespace std;

// concepts:
// design patterns. 
// async, fork join, producer-consumer, superloop

// 

void task(int id)
{
    cout<<"inside the task "<<id<<endl;
}

void addTask(int id, const int* a, const int* b, int* c)
{
    c[id] = a[id] + b[id];
}

int main()
{

    {
        std::vector<std::thread> threads;

        for(int i=0; i<100; i++)
        {   
            threads.emplace_back(task, i);
        }

        for(auto& t:threads) t.join();
    }

    {
        std::array<int, 100> a; 
        a.fill(10);
        std::array<int, 100> b; 
        b.fill(20);
        std::array<int, 100> c;
        c.fill(20);

        std::array<std::thread, 100> threads;

        for(int i=0; i<100; i++)
        {
            threads[i] = std::thread(addTask, i, a.data(), b.data(), c.data());
        }

        for(auto& t:threads) t.join();

        cout<<c[20]<<endl;
    }



    return 0;
}
