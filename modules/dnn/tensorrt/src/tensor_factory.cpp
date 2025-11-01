#include "tensor_factory.hpp"

Tensor TensorFactory::create(nvinfer1::ICudaEngine *engine, int index)
{
    const char *name = engine->getIOTensorName(index);
    nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
    auto dims = engine->getTensorShape(name);
    return Tensor(index, name, mode, dims);
}
