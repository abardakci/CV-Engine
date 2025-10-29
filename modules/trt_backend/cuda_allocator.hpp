#pragma once

#include <memory>

#include "cuda_runtime_api.h"
#include "helpers.hpp"

template <typename Derived>
class AllocatorBase
{
public:
    void allocate(void **ptr, size_t bytes);
    void free(void *ptr);
};


class DeviceAllocator : public AllocatorBase<DeviceAllocator>
{
public:
    void allocate_impl(void **ptr, size_t bytes);
    void free_impl(void *ptr);
};


class HostAllocator : public AllocatorBase<HostAllocator>
{
public:
    void allocate_impl(void **ptr, size_t bytes);
    void free_impl(void *ptr);
};
