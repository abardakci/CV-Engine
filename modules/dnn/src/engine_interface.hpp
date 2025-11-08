#pragma once

#include <string>
#include <vector>
#include "common.hpp"

enum class ShapeMode
{
    Static,
    Dynamic
};

class IEngine
{
public:
    virtual ~IEngine() = default;
    virtual void init(const std::string &engine_path) = 0;
    virtual void set_input(const float *input) = 0;
    virtual bool infer() = 0;
    virtual float *get_output() = 0;
    virtual bool bind_tensors() = 0;

    // Extend API
    virtual void set_input_shape(int idx, const tensor::Dims &shape)
    {
        throw std::runtime_error("Dynamic shapes not supported by this engine");
    }
};
