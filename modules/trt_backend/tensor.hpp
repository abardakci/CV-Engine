#pragma once

#include <NvInfer.h>
#include "cuda_runtime_api.h"

#include "cuda_buffer.hpp"
#include "helpers.hpp"

class Tensor
{
public:
    Tensor() = default;
    ~Tensor();

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    Tensor(int index, std::string name, nvinfer1::TensorIOMode mode, nvinfer1::Dims dimensions);

public:
    int index_ = -1;
    std::string name_;
    nvinfer1::TensorIOMode io_mode_;
    nvinfer1::Dims dims_;
    size_t elem_size_ = 0;
    CudaBuffer<float, HostAllocator> h_buffer;
    CudaBuffer<float, DeviceAllocator> d_buffer;
    
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
