

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <memory>
#include <thread>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#ifndef CHOICE
#define CHOICE 1
#endif

template <typename T>
using uptr = std::unique_ptr<T, std::function<void(T*)>>;


constexpr std::size_t def_size = 1 << 28;
constexpr int def_blck_sz = 32;



#define CHECK() \
    if (!check_all(size, Z.get(), X.get(), Y.get())) \
    { \
        std::cerr << "ERROR" << std::endl; \
        exit(EXIT_FAILURE); \
    }


template <typename T>
__device__ void calc(T *z, T *x, T *y)
{
    const auto xx = *x;
    const auto yy = *y;
    const auto zz = xx*xx + xx*yy + yy*yy;
    *z = zz;
}

template <typename T>
bool check(T *z, T *x, T *y)
{
    const auto xx = *x;
    const auto yy = *y;
    const auto zz = *z;
    auto test = (zz == xx*xx + xx*yy + yy*yy);
    return test;
}

template <typename T>
bool check_all(std::size_t N, T *Z, T *X, T *Y)
{
    for (std::size_t pos{}; pos!=N; ++pos)
    {
        if (!check(Z+pos,X+pos,Y+pos))
        {
            return false;
        }
    }
    return true;
}


template <typename T>
__device__ void op(std::size_t N, T *Z, T *X, T *Y)
{
    const std::size_t poolsz = (gridDim.x ? gridDim.x : 1) * blockDim.x;
    const std::size_t tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t pos = tIdx;
    while (pos < N)
    {
        // do op
        calc(Z+pos,X+pos,Y+pos);
        // next
        pos += poolsz;
    }
}


template <typename T>
__global__ void simple_kernel(std::size_t N, T *Z, T *X, T *Y)
{
    op(N, Z, X, Y);
}


template <typename T>
T *new_cuda(std::size_t size)
{
    const std::size_t totsz = size * sizeof(T);
    T *ptr = nullptr;
    assert(cudaSuccess == cudaMalloc((T**)&ptr, totsz));
    return ptr;
}

template <typename T>
T *new_host_alloc(std::size_t size)
{
    const std::size_t totsz = size * sizeof(T);
    T *ptr = nullptr;
    assert(cudaSuccess == cudaMallocHost((T**)&ptr, totsz));
    return ptr;
}


template <typename T>
T *new_host_alloc_mapped(std::size_t size)
{
    const std::size_t totsz = size * sizeof(T);
    T *ptr = nullptr;
    assert(cudaSuccess == cudaMallocHost((T**)&ptr, totsz, cudaHostAllocMapped));
    return ptr;
}

template <typename T>
T *new_host_alloc_unified(std::size_t size)
{
    const std::size_t totsz = size * sizeof(T);
    T *ptr = nullptr;
    assert(cudaSuccess == cudaMallocManaged((T**)&ptr, totsz));
    return ptr;
}

template <typename T>
T *host_to_dev_ptr(T *hptr)
{
    T *dptr = nullptr;
    assert(cudaSuccess == cudaHostGetDevicePointer((T**)&dptr, hptr, 0));
    return dptr;
}

template <typename T>
inline void cuda_allocate_default(std::size_t size, T *&dX, T *&dY, T *&dZ)
{
    const std::size_t totsz = size * sizeof(T);
    assert(cudaSuccess == cudaMalloc((T**)&dX, totsz));
    assert(cudaSuccess == cudaMalloc((T**)&dY, totsz));
    assert(cudaSuccess == cudaMalloc((T**)&dZ, totsz));
}

template <typename T>
void init(std::size_t size, T *X, T *Y, T *Z)
{
    for(std::size_t i = 0; i != size; i++)
    {
        X[i] = (T)10;
        Y[i] = (T)20;
    }
    memset(Z, 0, size*sizeof(T));
}

//
//  https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf
template <typename T>
void sol_mem_pageable(std::size_t size = def_size, int blck_sz = def_blck_sz)
{
    const std::size_t alloc_sz = size * sizeof(T);
    const std::size_t block_num = (int)std::min(((size + blck_sz-1)/blck_sz), (std::size_t)1 << 24);
    // define pointers
    std::unique_ptr<T> X(new T[size]), Y(new T[size]), Z(new T[size]);
    init(size, X.get(), Y.get(), Z.get());
    // instantiate device pointers
    auto del = [](T *ptr){ if(ptr) (void)cudaFree(ptr); };
    uptr<T> dX(new_cuda<T>(size), del);
    uptr<T> dY(new_cuda<T>(size), del);
    uptr<T> dZ(new_cuda<T>(size), del);
    // transfer from host to device memory
    cudaMemcpy(dX.get(), X.get(), alloc_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dY.get(), Y.get(), alloc_sz, cudaMemcpyHostToDevice);
    // invoke kernel
    simple_kernel<<<block_num, blck_sz>>>(size, dZ.get(), dX.get(), dY.get());
    // wait for termination
    cudaDeviceSynchronize();
    // transfer from device to host memory
    cudaMemcpy(Z.get(), dZ.get(), alloc_sz, cudaMemcpyDeviceToHost);

    // check everything ok
    CHECK();
    // memory is freed by destructors
}
template <typename T>
void sol_mem_pinned(std::size_t size = def_size, int blck_sz = def_blck_sz)
{
    const std::size_t alloc_sz = size * sizeof(T);
    const std::size_t block_num = (int)std::min(((size + blck_sz-1)/blck_sz), (std::size_t)1 << 24);
    // define pointers
    uptr<T> X(new_host_alloc<T>(size), cudaFreeHost), Y(new_host_alloc<T>(size), cudaFreeHost), Z(new_host_alloc<T>(size), cudaFreeHost);
    init(size, X.get(), Y.get(), Z.get());
    // instantiate device pointers
    auto del = [](T *ptr){ if(ptr) (void)cudaFree(ptr); };
    uptr<T> dX(new_cuda<T>(size), del);
    uptr<T> dY(new_cuda<T>(size), del);
    uptr<T> dZ(new_cuda<T>(size), del);
    // transfer from host to device memory
    cudaMemcpy(dX.get(), X.get(), alloc_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dY.get(), Y.get(), alloc_sz, cudaMemcpyHostToDevice);
    // invoke kernel
    simple_kernel<<<block_num, blck_sz>>>(size, dZ.get(), dX.get(), dY.get());
    // wait for termination
    cudaDeviceSynchronize();
    // transfer from device to host memory
    cudaMemcpy(Z.get(), dZ.get(), alloc_sz, cudaMemcpyDeviceToHost);

    // check everything ok
    CHECK();
}
//  https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
template <typename T>
void sol_mem_mapped(std::size_t size = def_size, int blck_sz = def_blck_sz)
{
    assert(cudaSuccess == cudaSetDeviceFlags(cudaDeviceMapHost));
    const std::size_t block_num = (int)std::min(((size + blck_sz-1)/blck_sz), (std::size_t)1 << 24);
    // define pointers
    uptr<T> X(new_host_alloc_mapped<T>(size), cudaFreeHost), Y(new_host_alloc_mapped<T>(size), cudaFreeHost), Z(new_host_alloc_mapped<T>(size), cudaFreeHost);
    init(size, X.get(), Y.get(), Z.get());
    // instantiate device pointers
    T* dX = host_to_dev_ptr(X.get());
    T* dY = host_to_dev_ptr(Y.get());
    T* dZ = host_to_dev_ptr(Z.get());
    // NOT REQUIRED: transfer from host to device memory
    // invoke kernel
    simple_kernel<<<block_num, blck_sz>>>(size, dZ, dX, dY);
    // wait for termination
    cudaDeviceSynchronize();
    // NOT REQUIRED: transfer from device to host memory
    // check everything ok
    CHECK();
}
template <typename T>
void sol_mem_unified(std::size_t size = def_size, int blck_sz = def_blck_sz)
{
    const std::size_t block_num = (int)std::min(((size + blck_sz-1)/blck_sz), (std::size_t)1 << 24);
    // define pointers
    uptr<T> X(new_host_alloc_unified<T>(size), cudaFree), Y(new_host_alloc_unified<T>(size), cudaFree), Z(new_host_alloc_unified<T>(size), cudaFree);
    init(size, X.get(), Y.get(), Z.get());
    // NOT REQUIRED:  instantiate device pointers
    // NOT REQUIRED: transfer from host to device memory
    // invoke kernel
    simple_kernel<<<block_num, blck_sz>>>(size, Z.get(), X.get(), Y.get());
    // wait for termination
    cudaDeviceSynchronize();
    // NOT REQUIRED: transfer from device to host memory
    // check everything ok
    CHECK();
}


template <typename FUN>
void measurer(FUN f)
{
    const auto start = std::chrono::high_resolution_clock::now();
    f();
    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = stop - start;
    std::cout << "Duration = " << duration.count() << std::endl;
}


int main(int argc, char const *argv[])
{
    std::function<void(void)> f = [](){
#if CHOICE == 1
#pragma message "Using: sol_mem_pageable"
        sol_mem_pageable<int>();
#elif CHOICE == 2
#pragma message "Using: sol_mem_pinned"
        sol_mem_pinned<int>();
#elif CHOICE == 3
#pragma message "Using: sol_mem_mapped"
        sol_mem_mapped<int>();
#elif CHOICE == 4
#pragma message "Using: sol_mem_unified"
        sol_mem_unified<int>();
#else
  #error Unsupported choice setting
#endif
    };
    measurer(f);
    return 0;
}


