#include "tensor.hpp"

using namespace nvinfer1;

Tensor::Tensor(std::string name, TensorIOMode mode, nvinfer1::Dims dims)
{
    std::cout << "[Tensor(name,mode,dims)]" << name << "\n";
    name_ = name;
    mode_ = mode;
    dims_ = dims;
    elem_size_ = calc_elem_size(dims);
    CUDA_CHECK(cudaMalloc((void**)&d_buffer_, elem_size_ * sizeof(float)));

    if (mode == TensorIOMode::kINPUT)
        h_buffer_ = nullptr; // will be set by set_input(T*)
    else if (mode == TensorIOMode::kOUTPUT)
        h_buffer_ = new float[elem_size_];
}

Tensor::~Tensor()
{
    if (d_buffer_)
        cudaFree((void *)d_buffer_);
    if (mode_ == TensorIOMode::kOUTPUT)
        delete[] h_buffer_;
}
