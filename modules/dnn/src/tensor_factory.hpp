#pragma once

#include "tensor.hpp"

class TensorFactory
{
public:
    Tensor create(void *engine, std::shared_ptr<IAllocatorBase> allocator, int index);

};

extern TensorFactory tensorFactory;
