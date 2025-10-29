#include "cuda_buffer.hpp"

template<typename T, typename Allocator>
CudaBuffer<T, Allocator>::~CudaBuffer()
{
    allocator_->free();
}

template<typename T, typename Allocator>
void CudaBuffer<T, Allocator>::allocate(const size_t size)
{
    size_ = size;
    allocator_->allocate(size);
}

template<typename T, typename Allocator>
T* CudaBuffer<T, Allocator>::get() noexcept
{
    return buffer;
}

template<typename T, typename Allocator>
size_t CudaBuffer<T, Allocator>::size() noexcept
{
    return size_;
}
