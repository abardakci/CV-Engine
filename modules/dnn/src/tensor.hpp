#pragma once

#include <NvInfer.h>
#include "cuda_runtime_api.h"

#include "cuda_buffer.hpp"
#include "helpers.hpp"

class Tensor
{
public:
    ~Tensor();

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    Tensor(int index, std::string name, nvinfer1::TensorIOMode mode, nvinfer1::Dims dimensions, std::shared_ptr<IAllocatorBase> allocator_);

public:
    int index_ = -1;
    std::string name_;
    nvinfer1::TensorIOMode io_mode_;
    nvinfer1::Dims dims_;
    size_t elem_size_ = 0;
    std::shared_ptr<IAllocatorBase> allocator_;
    CudaBuffer buffer_;
};
