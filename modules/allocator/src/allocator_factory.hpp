#pragma once

#include "base_allocator.hpp"
#include <memory>

class AllocatorFactory
{
public:
    std::unique_ptr<IAllocatorBase> create(backend::Tag tag);
};
