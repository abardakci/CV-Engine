#include "cuda_allocator.hpp"

void CudaAllocator::allocate_impl(void **ptr, size_t bytes)
{
    BackendTrait::malloc(ptr, bytes);
}

void CudaAllocator::free_impl(void *ptr)
{
    BackendTrait::free(ptr);
}

void CudaAllocator::allocate_host_impl(void **ptr, size_t bytes)
{
    BackendTrait::malloc_host(ptr, bytes);
}

void CudaAllocator::free_host_impl(void *ptr)
{
    BackendTrait::free_host(ptr);
}

void CudaAllocator::memcpy_impl(void *dst, void *src, size_t bytes, CopyMode copy_mode)
{
    cudaMemcpyKind cuda_copy_mode = to_cuda_copy_mode(copy_mode);
    BackendTrait::memcpy(dst, src, bytes, cuda_copy_mode);
}

void CudaAllocator::memcpy_async_impl(void *dst, void *src, size_t bytes, CopyMode copy_mode) {}
