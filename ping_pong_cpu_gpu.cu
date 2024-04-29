
#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <type_traits>


template <typename T>
using uptr = std::unique_ptr<T, std::function<void(T*)>>;


template <typename T>
T *new_host_alloc_mapped(std::size_t size)
{
    const std::size_t totsz = size * sizeof(T);
    T *ptr = nullptr;
    assert(cudaSuccess == cudaMallocHost((T**)&ptr, totsz, cudaHostAllocMapped));
    return ptr;
}

template <typename T>
T *host_to_dev_ptr(T *hptr)
{
    using TT = typename std::remove_volatile<T>::type;
    TT *dptr = nullptr;
    assert(cudaSuccess == cudaHostGetDevicePointer((TT**)&dptr, (TT*)hptr, 0));
    return dptr;
}

/**
 * guard: variable observed by both CPU and GPU to sync work
 *
 */
//template <typename T>
__global__ void incrementer(volatile int *guard, volatile int*data, std::size_t values, std::size_t reps)
{
    int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    bool isMaster = tIdx == 0;
    // for group synchornization
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
//    if (isMaster) {
//        printf("Helo from Master\n");
//    }

    for (std::size_t i{}; i!=reps; ++i)
    {
        // inc values
        if (tIdx < values) {
            data[tIdx] *= 2;
        }
//        if (isMaster) {
//            printf("Master before g.sync()\n");
//        }
        // sync grid
        g.sync();
//        if (isMaster) {
//            printf("Master after g.sync()\n");
//        }

        // only for master
        if (isMaster) {
            // change value seen by CPU
            // from even to odd: unblock CPU
            *guard += 1;
            //printf("master guard: %d\n", *guard);
        }
        // if at end break
        if (i+1 == reps) {
            break;
        }
        // wait for update from CPU
        if (isMaster)
        {
            // wait until guard become even
            while ((*guard) & 1)
            {}
        }
        // sync grid
        g.sync();
    }
}



void ping_pong(
    std::size_t reps,
    std::size_t values
)
{
    constexpr std::size_t blk_sz = 32ul;
    const std::size_t blk_num = (values+blk_sz-1)/blk_sz;

    // allocate 0-copy memory for guard
    assert(cudaSuccess == cudaSetDeviceFlags(cudaDeviceMapHost));
    uptr<volatile int> guard_holder((volatile int*)new_host_alloc_mapped<int>(1ul), (void(*)(volatile int*))cudaFreeHost);
    volatile int *const guard_cpuptr = guard_holder.get();
    volatile int *const guard_gpuptr = host_to_dev_ptr(guard_cpuptr);

    // allocate 0-copy memory for data
    uptr<volatile int> data_holder((volatile int*)new_host_alloc_mapped<int>(values), (void(*)(volatile int*))cudaFreeHost);
    volatile int *const data_cpuptr = data_holder.get();
    volatile int *const data_gpuptr = host_to_dev_ptr(data_cpuptr);

    std::cerr << "GUARD:\t" << (void*)guard_cpuptr << "\t" << (void*)guard_gpuptr << std::endl;
    std::cerr << "DATA: \t" << (void*)data_cpuptr << "\t" << (void*)data_gpuptr << std::endl;

    for (std::size_t j{}; j!=values; ++j) {
        data_cpuptr[j] = j;
    }

    std::cerr << "START\n";
    const auto start = std::chrono::high_resolution_clock::now();
    //incrementer<<<blk_num, blk_sz>>>(guard_gpuptr, data_gpuptr, values, reps);
    void *kernelArgs[] = {
        (void*)&guard_gpuptr,
        (void*)&data_gpuptr,
        (void*)&values,
        (void*)&reps
    };
    cudaLaunchCooperativeKernel((void*)incrementer, dim3(blk_num,1,1), dim3(blk_sz,1,1), kernelArgs);

    std::vector<std::size_t> fails(reps);
    for (std::size_t i{}; i!=reps; ++i) {
        // wait for GPU op
        std::size_t fail = 0;
        while (!((*guard_cpuptr) & 1))
        {
            ++fail;
//            if (fail%50000000 == 0) {
//                std::cerr << "fails:\t" << fail << "\tguard:\t" << *guard_cpuptr << std::endl;
//            }
        }
        fails[i] = fail;
        // do op on CPU
        for (std::size_t j{}; j!=values; ++j) {
            data_cpuptr[j] -= j;
        }
        // from odd to even: unblock GPU
        *guard_cpuptr += 1;
    }
    // termination
    cudaDeviceSynchronize();

    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = stop - start;
    std::cout << "Inner duration = " << duration.count() << std::endl;

    for (std::size_t i{}; i!=reps; ++i) {
        std::cout << i << ")\t" << fails[i] << std::endl;
    }
    int bugs = 0;
    for (std::size_t j{}; j!=values; ++j) {
        if (data_cpuptr[j] != j) {
            ++bugs;
            //std::cout << "data_cpuptr[" << j << "] = " << data_cpuptr[j] << std::endl;
        }
    }
    std::cerr << "bugs = " << bugs << std::endl;
}


int main(int argc, char const *argv[])
{
    std::size_t reps = 100;
    std::size_t values = 10000000;

    bool fail = false;

    if (argc > 1) {
        // get reps
        reps = std::stoull(argv[1]);
        if (errno != 0 || reps <= 0) {
            fail = true;
        }
    }
    if (argc > 2) {
        // get values
        values = std::stoull(argv[2]);
        if (errno != 0 || values <= 0) {
            fail = true;
        }
    }
    if (argc > 3) {
        fail = true;
    }
    if (fail) {
        std::cerr << "Bad arguments, at most 2 numeric values expected: (reps, N)" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << argv[0] << "\t" << reps << "\t" << values << "\n";

    std::cout << "Helo!" << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();

    //std::this_thread::sleep_for(std::chrono::seconds(1));
    ping_pong(reps, values);

    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = stop - start;
    std::cout << "Duration = " << duration.count() << std::endl;

    return 0;
}



