#pragma once

#include <string>
#include <vector>

// // FP16 or FP32
// template <typename T>
// concept TensorDtype = std::same_as<T, float> || std::same_as<T, uint16_t>;
// enum class Precision { FP16, FP32 };

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
    virtual void set_input_dynamic(const float *input, const std::vector<int>& shape) = 0;
    virtual bool infer() = 0;
    virtual float *get_output() = 0;
    virtual std::vector<int> get_output_dims() const = 0;
};
