#pragma once

#include "base_factory.hpp"

class CudaBuffer
{
public:
    CudaBuffer(); 
    ~CudaBuffer();

    // Non-copyable
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    // Non-Movable for now
    CudaBuffer(CudaBuffer &&other) noexcept;
    CudaBuffer &operator=(CudaBuffer &&) noexcept;

    void setAllocator(std::shared_ptr<IAllocatorBase> allocator);
    void allocate(size_t size);

public:
    std::shared_ptr<IAllocatorBase> allocator_;
    float *d_data_;
    float *h_data_;
    size_t size_;
};
