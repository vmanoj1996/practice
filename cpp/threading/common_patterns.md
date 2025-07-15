# Questions?

- Mutex side effect - >raise the priority of current thread/task

## OS VS RTOS

### OS

Process → Contains multiple threads
Thread  → Lightweight execution unit within process

#### RTOS
Task = Thread = Basic execution unit

1. RTOS typically runs the highest priority READY task.
2. Mutex locks rises the priority of task 


READY    - Can run immediately
RUNNING  - Currently executing  
BLOCKED  - Waiting for resource/event
SUSPENDED - Manually suspended


# Common Multithreading Design Patterns:
1. Producer-Consumer

Producers create data, consumers process it
Uses queues/buffers between threads

cppstd::queue<Task> taskQueue;
std::mutex queueMutex;
std::condition_variable cv;


2. Thread Pool

Fixed number of worker threads
Reuse threads instead of creating/destroying
Submit tasks to shared work queue

3. Reader-Writer Lock

Multiple readers OR single writer
Optimizes read-heavy workloads

cppstd::shared_mutex rwMutex;
// Multiple readers: shared_lock
// Single writer: unique_lock


4. Future/Promise

Async task execution with result retrieval

cppauto future = std::async(std::launch::async, computeTask);
auto result = future.get();  // Blocks until done

5. Actor Model

Isolated actors communicate via messages
No shared state, only message passing

6. Pipeline

Chain of processing stages
Each stage runs in separate thread
Data flows through stages

7. Lock-Free Programming

Use atomic operations instead of locks

cppstd::atomic<int> counter{0};
counter.fetch_add(1);  // Thread-safe increment
8. Master-Worker

Master distributes work to worker threads
Workers report results back to master