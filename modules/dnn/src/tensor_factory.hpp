#pragma once

#include <NvInfer.h>
#include <tensor.hpp>

class TensorFactory
{
public:
    Tensor create(nvinfer1::ICudaEngine *engine, std::shared_ptr<IAllocatorBase> allocator, int index);
};

extern TensorFactory tensorFactory;
