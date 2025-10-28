#pragma once

#include <NvInfer.h>
#include "cuda_runtime_api.h"

#include "helpers.hpp"

class Tensor
{
public:
    Tensor() = default;
    Tensor(std::string name, nvinfer1::TensorIOMode mode, nvinfer1::Dims dimensions);
    ~Tensor();

public:
    std::string name_;
    nvinfer1::TensorIOMode mode_;
    nvinfer1::Dims dims_;
    size_t elem_size_;
    float *h_buffer_;
    float *d_buffer_;
};

inline size_t calc_elem_size(nvinfer1::Dims dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        const int dim = dims.d[i];
        if (dim == 0)
            continue;
        if (dim == -1)
            return -1;

        size *= dims.d[i];
    }

    return size;
}
