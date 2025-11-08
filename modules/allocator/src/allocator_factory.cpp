#include "allocator_factory.hpp"
#include "cuda_allocator.hpp"

#include <memory>

std::shared_ptr<IAllocatorBase> AllocatorFactory::create(backend::Tag type)
{
    static std::shared_ptr<IAllocatorBase> cuda_singleton = std::make_shared<CudaAllocator>();

    switch(type)
    {
        case backend::Tag::CUDA:
            return cuda_singleton;
        default:
            throw("NOT IMPLEMENTED ERROR");
    }
}
