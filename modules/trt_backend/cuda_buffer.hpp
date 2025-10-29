#pragma once

#include "cuda_allocator.hpp"

template <typename T, typename Allocator>
class CudaBuffer
{
public:
    CudaBuffer() = default; 
    ~CudaBuffer();

    // Non-copyable
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    // Non-Movable for now
    CudaBuffer(CudaBuffer &&other) noexcept = delete;
    CudaBuffer &operator=(CudaBuffer &&) noexcept = delete;

    void CudaBuffer::allocate(const size_t size);
    T *get() noexcept;
    size_t size();

private:
    Allocator allocator_;
    T *buffer_;
    size_t size_;
};
