#pragma once
#include <cstddef>

namespace backend
{
    enum class Tag
    {
        CUDA,
        ROCM
    };

    enum class CopyMode
    {
        DeviceToDevice,
        DeviceToHost,
        HostToHost,
        HostToDevice
    };

    template <backend::Tag BackendTag>
    struct Traits;
};

class IAllocatorBase
{
public:
    virtual ~IAllocatorBase() = default;

    virtual void allocate(void **ptr, size_t bytes) = 0;
    virtual void free(void *ptr) = 0;
    virtual void memcpy(void *dst, void *src, size_t bytes, backend::CopyMode copy_mode) = 0;
    virtual void memcpy_async(void *dst, void *src, size_t bytes, backend::CopyMode copy_mode, void *stream = nullptr) = 0;
};
