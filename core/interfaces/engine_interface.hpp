#pragma once

#include <memory>
#include <array>
#include <string>

static constexpr int N_MAX_INPUT = 1;
static constexpr int N_MAX_OUTPUT = 1;

class IEngine
{
public:
    virtual ~IEngine() = default;
    virtual void init(const std::string &engine_path) = 0;
    virtual bool infer(const std::array<float *, N_MAX_INPUT> inputs,
                       const std::array<float *, N_MAX_OUTPUT> outputs) = 0;
};
