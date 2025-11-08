#pragma once

#include <array>
#include <memory>

namespace tensor
{
    enum class IOMode
    {
        InputTensor,
        OutputTensor
    };
    enum class ShapeMode
    {
        Dynamic,
        Static
    };
    struct Dims
    {
        std::array<size_t, 8> d;
        Dims() { d.fill(0); }
    };
}