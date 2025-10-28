#pragma once

#include <memory>
#include <array>
#include <string>
#include <type_traits>
#include <stdfloat>

// // FP16 or FP32
// template <typename T>
// concept TensorDtype = std::same_as<T, float> || std::same_as<T, uint16_t>;
// enum class Precision { FP16, FP32 };

class IEngine
{
public:
    virtual ~IEngine() = default;
    virtual void init(const std::string &engine_path) = 0;
    virtual void set_input(float *input) = 0;
    virtual bool infer() = 0;
    virtual float *get_output() = 0;
};
