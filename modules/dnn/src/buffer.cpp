#include "cuda_buffer.hpp"

CudaBuffer::CudaBuffer(CudaBuffer &&other) noexcept
    : d_data_(other.d_data_), h_data_(other.h_data_), size_(other.size_), allocator_(std::move(other.allocator_))
{
    other.d_data_ = nullptr;
    other.h_data_ = nullptr;
    other.size_ = 0;
}

CudaBuffer &CudaBuffer::operator=(CudaBuffer &&other) noexcept
{
    if (this != &other)
    {
        d_data_ = other.d_data_;
        h_data_ = other.h_data_;
        size_ = other.size_;
        other.d_data_ = nullptr;
        other.h_data_ = nullptr;
        other.size_ = 0;
    }
}

void CudaBuffer::setAllocator(std::shared_ptr<IAllocatorBase> allocator)
{
    allocator_ = allocator;
}

CudaBuffer::~CudaBuffer()
{
    if (d_data_)
        allocator_->free(d_data_);
    if (h_data_)
        allocator_->free_host(h_data_);
}

void CudaBuffer::allocate(size_t size)
{
    size_ = size;
    allocator_->allocate((void **)&d_data_, size_);
    allocator_->allocate_host((void **)&h_data_, size_);
}
