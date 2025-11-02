#include "tensor_factory.hpp"

Tensor TensorFactory::create(nvinfer1::ICudaEngine *engine, std::shared_ptr<IAllocatorBase> allocator, int index)
{
    const char *name = engine->getIOTensorName(index);
    nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
    auto dims = engine->getTensorShape(name);
    return Tensor(index, name, mode, dims, allocator);
}
