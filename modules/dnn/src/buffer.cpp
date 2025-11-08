#include "buffer.hpp"

Buffer::Buffer(Buffer &&other) noexcept
    : d_data_(other.d_data_),
      h_data_(other.h_data_),
      size_(other.size_),
      allocator_(other.allocator_)
{
    other.d_data_ = nullptr;
    other.h_data_ = nullptr;
    other.size_ = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept
{
    if (this != &other)
    {
        d_data_ = other.d_data_;
        h_data_ = other.h_data_;
        size_ = other.size_;
        allocator_ = other.allocator_;

        other.d_data_ = nullptr;
        other.h_data_ = nullptr;
        other.size_ = 0;
    }

    return *this;
}

Buffer::~Buffer()
{
    release();
}

void Buffer::release()
{
    if (d_data_)
        allocator_->free(d_data_);
    if (h_data_)
        allocator_->free_host(h_data_);
    size_ = 0;
}

void Buffer::allocate(size_t size)
{
    size_ = size;
    allocator_->allocate((void **)&d_data_, size_);
    allocator_->allocate_host((void **)&h_data_, size_);
}

void Buffer::reallocate(size_t size)
{
    release();
    allocate(size);
}
