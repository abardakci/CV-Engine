#pragma once

#include "tensor.hpp"

class TensorFactory
{
public:
    Tensor create(void *engine, std::shared_ptr<IAllocatorBase> allocator, int index);

private:
    backend::Tag BackendTag;
};

extern TensorFactory tensorFactory;
