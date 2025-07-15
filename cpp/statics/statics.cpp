/*
Memory layout in programs

High Memory Address
┌─────────────────┐
│     Stack       │ ← Local variables, function calls
│        ↓        │   (grows downward)
├─────────────────┤
│                 │
│   Free Space    │
│                 │
├─────────────────┤
│     Heap        │ ← Dynamic allocation (new/malloc)
│        ↑        │   (grows upward)
├─────────────────┤
│  Uninitialized  │ ← Uninitialized globals/static (.bss)
│     Data        │   (zero-initialized). This section is not stored inside the executable file (only size spec is stored->reduces load time). It is initialized in ram to 0 directly.
├─────────────────┤
│   Initialized   │ ← Initialized globals/static (.data)
│     Data        │
├─────────────────┤
│   Text/Code     │ ← Program instructions (.text)
│    (Read-only)  │
└─────────────────┘
Low Memory Address

Internally the statics becomes something like:
cpp// file1.cpp - compiler creates unique symbol
int _file1_counter = 10;

// file2.cpp - compiler creates different symbol  
int _file2_counter = 20;

*/

#include <iostream>
#include <vector>
#include "statics.hpp"

int bla[100] = {0};

static int counter = 0;

// allocate the storage in this cpp file for that static object
int Person::population = 0;

int changeSomething();

int main()
{
    std::vector<Person> bangalore(1000);

    // access controls still apply
    std::cout<<Person().population<<" "<<
                Person::population<<" "<<
                Person::getPopulation()<<" "<<
                Person().getPopulationInstance()<<" "<<
                Person().getPopulation()<<std::endl;

    std::cout<<++counter<<std::endl;

    changeSomething();
    std::cout<<bla[0]<<std::endl;

    return 0;
}
