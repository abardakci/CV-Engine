#include "base_allocator/factory.hpp"

#include "cuda_alloc/allocator.hpp"

#include <memory>

std::unique_ptr<IAllocatorBase> AllocatorFactory::create(backend::Tag tag)
{
    switch (tag)
    {
    case backend::Tag::CUDA:
        return std::make_unique<CudaAllocator>();

    // case backend::Tag::ROCM:
    //     return std::make_unique<RocmAllocator>();

    default:
        break;
    }
}
