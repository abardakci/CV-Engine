#include "cuda_allocator.hpp"

template <typename Derived>
void AllocatorBase<Derived>::allocate(void **ptr, size_t bytes)
{
    static_cast<Derived *>(this)->allocate_impl(ptr, bytes);
}

template <typename Derived>
void AllocatorBase<Derived>::free(void *ptr)
{
    static_cast<Derived *>(this)->free_impl(ptr);
}


void DeviceAllocator::allocate_impl(void **ptr, size_t bytes)
{
    CUDA_CHECK(cudaMalloc(ptr, bytes));
}

void DeviceAllocator::free_impl(void *ptr)
{
    CUDA_CHECK(cudaFree(ptr));
}


void HostAllocator::allocate_impl(void **ptr, size_t bytes)
{
    CUDA_CHECK(cudaMallocHost(ptr, bytes));
}

void HostAllocator::free_impl(void *ptr)
{
    CUDA_CHECK(cudaFreeHost(ptr));
}
