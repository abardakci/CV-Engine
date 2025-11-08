#include "tensor.hpp"

Tensor::Tensor(int index, std::string name, tensor::IOMode mode, tensor::Dims dims, backend::Tag backend_tag)
    : index_(index),
      name_(std::move(name)),
      io_mode_(mode),
      dims_(dims),
      buffer_(Buffer(backend_tag))
{
    std::optional<size_t> opt_size = calc_elem_size(dims_);

    if (!opt_size.has_value()) // dynamic shape
    {
        shape_mode_ = tensor::ShapeMode::Dynamic;
        elem_size_ = 0;
    }
    else
    {
        shape_mode_ = tensor::ShapeMode::Static;
        elem_size_ = *opt_size;
        if (elem_size_ > 0)
            buffer_.allocate(elem_size_ * sizeof(float));
    }
}

Tensor::Tensor(Tensor &&other) noexcept
    : index_(other.index_),
      name_(std::move(other.name_)),
      io_mode_(other.io_mode_),
      dims_(other.dims_),
      elem_size_(other.elem_size_),
      buffer_(std::move(other.buffer_)), // move CudaBuffer
      shape_mode_(other.shape_mode_)
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
        shape_mode_ = other.shape_mode_;

        other.index_ = -1;
        other.elem_size_ = 0;
    }

    return *this;
}
