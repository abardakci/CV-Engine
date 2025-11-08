#include "tensor.hpp"

static size_t calc_elem_size(tensor::Dims dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.d.size(); ++i)
    {
        const int dim = dims.d[i];
        if (dim == 0)
            continue;
        if (dim == -1)
            return -1;

        size *= dim;
    }

    return size;
}

Tensor::Tensor(int index, std::string name, tensor::IOMode mode, tensor::Dims dims, backend::Tag backend_tag)
    : index_(index),
      name_(std::move(name)),
      io_mode_(mode),
      dims_(dims),
      buffer_(backend_tag)
{
    elem_size_ = calc_elem_size(dims_);
    buffer_.allocate(elem_size_ * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept
    : index_(other.index_),
      name_(std::move(other.name_)),
      io_mode_(other.io_mode_),
      dims_(other.dims_),
      elem_size_(other.elem_size_),
      buffer_(std::move(other.buffer_)) // move CudaBuffer
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
        buffer_ = std::move(other.buffer_); // move CudaBuffer

        other.index_ = -1;
        other.elem_size_ = 0;
    }

    return *this;
}
