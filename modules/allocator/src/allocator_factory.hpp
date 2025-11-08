#pragma once

#include "base_allocator.hpp"
#include <memory>

class AllocatorFactory
{
public:
    static std::shared_ptr<IAllocatorBase> create(backend::Tag type);
};
