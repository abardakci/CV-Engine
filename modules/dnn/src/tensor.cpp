#include "tensor.hpp"

using namespace nvinfer1;

Tensor::Tensor(int index, std::string name, nvinfer1::TensorIOMode mode, nvinfer1::Dims dimensions, std::shared_ptr<IAllocatorBase> allocator)
    : index_(index), name_(name), io_mode_(mode), dims_(dimensions), allocator_(allocator)
{
    elem_size_ = calc_elem_size(dims_);
    buffer_.allocator_ = allocator;
    buffer_.allocate(elem_size_ * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept
    : index_(other.index_),
      name_(std::move(other.name_)),
      io_mode_(other.io_mode_),
      dims_(other.dims_),
      elem_size_(other.elem_size_),
      buffer_(std::move(other.buffer_))  // move CudaBuffer
{
    other.index_ = -1;
    other.elem_size_ = 0;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        index_ = other.index_;
        name_ = std::move(other.name_);
        io_mode_ = other.io_mode_;
        dims_ = other.dims_;
        elem_size_ = other.elem_size_;
        buffer_ = std::move(other.buffer_);  // move CudaBuffer

        other.index_ = -1;
        other.elem_size_ = 0;
    }
    
    return *this;
}
