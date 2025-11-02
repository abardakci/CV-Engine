#include "allocator_factory.hpp"
#include "cuda_allocator.hpp"

#include <memory>

std::unique_ptr<IAllocatorBase> AllocatorFactory::create(backend::Tag tag)
{
    switch (tag)
    {
    case backend::Tag::CUDA:
        return std::make_unique<CudaAllocator>();

    default:
        break;
    }
}
