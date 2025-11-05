#pragma once

#include "allocator_interface.hpp"
#include <memory>

template <typename Derived, backend::Tag BackendTag>
class BaseAllocator : public IAllocatorBase
{
public:
    using BackendTrait = backend::Traits<BackendTag>;
    using CopyMode = typename backend::CopyMode;
    using StreamType = typename backend::Traits<BackendTag>::StreamType;

    void allocate(void **ptr, size_t bytes) override
    {
        static_cast<Derived *>(this)->allocate_impl(ptr, bytes);
    }
    
    void allocate_host(void **ptr, size_t bytes) override
    {
        static_cast<Derived *>(this)->allocate_host_impl(ptr, bytes);
    }

    void free(void *ptr) override
    {
        static_cast<Derived *>(this)->free_impl(ptr);
    }

    void free_host(void *ptr) override
    {
        static_cast<Derived *>(this)->free_host_impl(ptr);
    }

    void memcpy(void *dst, void *src, size_t bytes, CopyMode copy_mode) override
    {
        static_cast<Derived *>(this)->memcpy_impl(dst, src, bytes, copy_mode);
    }

    void memcpy_async(void *dst, void *src, size_t bytes, CopyMode copy_mode, backend::StreamHandle stream) override
    {
        static_cast<Derived *>(this)->memcpy_async_impl(dst, src, bytes, copy_mode, stream);
    }
};
