#pragma once

#include <array>
#include <string>
#include <memory>

#include "buffer.hpp"

namespace tensor
{
    enum class IOMode { InputTensor, OutputTensor };
    struct Dims
    {
        std::array<size_t, 8> d;
        Dims() { d.fill(0); } 
    };
}

class Tensor
{
public:
    Tensor() = default;
    ~Tensor() = default;

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    Tensor(int index, std::string name, tensor::IOMode mode, tensor::Dims dims, std::shared_ptr<IAllocatorBase> allocator);

public:
    int index_ = -1;
    std::string name_;
    tensor::IOMode io_mode_;
    tensor::Dims dims_;
    size_t elem_size_ = 0;
    std::shared_ptr<IAllocatorBase> allocator_;
    Buffer buffer_;
};
