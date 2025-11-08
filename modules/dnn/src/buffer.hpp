#pragma once

#include "allocator_factory.hpp"

class Buffer
{
public:
    Buffer() = default;
    Buffer(backend::Tag tag) { allocator_ = AllocatorFactory::create(tag); };
    ~Buffer();

    // Non-copyable
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    // Non-Movable for now
    Buffer(Buffer &&other) noexcept;
    Buffer &operator=(Buffer &&) noexcept;

    void allocate(size_t size);
    void release();
    void reallocate(size_t size);

public:
    std::shared_ptr<IAllocatorBase> allocator_;
    float *d_data_;
    float *h_data_;
    size_t size_;
};
