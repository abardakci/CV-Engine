#include "tensor_factory.hpp"

#include <NvInfer.h>

Tensor TensorFactory::create(void *engine, backend::Tag backend_tag, int index)
{
    auto _engine = static_cast<nvinfer1::ICudaEngine*>(engine);

    const char *name = _engine->getIOTensorName(index);
    nvinfer1::TensorIOMode mode = _engine->getTensorIOMode(name);

    tensor::IOMode _mode;
    if (mode == nvinfer1::TensorIOMode::kINPUT) _mode = tensor::IOMode::InputTensor;
    else if (mode == nvinfer1::TensorIOMode::kOUTPUT) _mode = tensor::IOMode::OutputTensor;

    auto dims = _engine->getTensorShape(name);
    tensor::Dims _dims;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        _dims.d[i] = dims.d[i];
    }

    return Tensor(index, name, _mode, _dims, backend_tag);
}
