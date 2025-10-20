#pragma once

#include <memory>

class IEngine {
public:
    virtual ~IEngine() = default;
    virtual void initialize(const std::string&) = 0; 
    virtual int infer(float* input, float* output) = 0;
    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;
};
