#pragma once

#include <array>
#include <string>
#include <memory>
#include <optional>

#include "buffer.hpp"
#include "common.hpp"

inline std::optional<size_t> calc_elem_size(const tensor::Dims& dims)
{
    size_t size = 1;
    for (int dim : dims.d)
    {
        if (dim < 0)
            return std::nullopt; // dynamic
        if (dim == 0)
            continue;            // empty tensor
        size *= static_cast<size_t>(dim);
    }
    return size;
}

class Tensor
{
public:
    Tensor() = default;
    Tensor(tensor::ShapeMode mode) : shape_mode_(mode) {};
    ~Tensor() = default;

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    Tensor(int index, std::string name, tensor::IOMode mode, tensor::Dims dims, backend::Tag backend_tag);
    void set_shape(const tensor::Dims &new_dims)
    {
        auto data = calc_elem_size(new_dims);
        size_t new_bytes = data.value() * sizeof(float);

        if (buffer_.size_ < new_bytes)
            buffer_.reallocate(new_bytes);

        dims_ = new_dims;
    }

public:
    int index_ = -1;
    std::string name_;
    tensor::IOMode io_mode_;
    tensor::Dims dims_;
    size_t elem_size_ = 0;
    std::shared_ptr<IAllocatorBase> allocator_;
    Buffer buffer_;
    tensor::ShapeMode shape_mode_;
};
