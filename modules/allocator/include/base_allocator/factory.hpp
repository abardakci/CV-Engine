#include "allocator.hpp"

class AllocatorFactory
{
public:
    std::unique_ptr<IAllocatorBase> create(backend::Tag tag);
};
