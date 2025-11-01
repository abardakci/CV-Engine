#include <iostream>
#include "base_allocator/factory.hpp"

int main()
{
    std::cout << "[TEST]\n";
    std::unique_ptr<IAllocatorBase> allocator = AllocatorFactory().create(backend::Tag::CUDA);

    float* ptr = nullptr;
    size_t size = 128;
    allocator->allocate((void**) &ptr, 128);
    allocator->free(ptr);
}