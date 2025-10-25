#pragma once

#include <memory>
#include <vector>

static constexpr int N_MAX_INPUT = 1;
static constexpr int N_MAX_OUTPUT = 1;

class IEngine {
public:
    virtual ~IEngine() = default;
    virtual void initialize(const std::string&) = 0; 
    virtual int infer(std::array<float*, N_MAX_INPUT> inputs, std::array<float*, N_MAX_OUTPUT> outputs) = 0;
    
};
