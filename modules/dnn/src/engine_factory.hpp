#pragma once

#include <memory>
#include "engine_interface.hpp"

enum class EngineType
{
    TensorRT,
    OnnxRuntime
};

class EngineFactory
{
public:
    ~EngineFactory() = default;
    std::unique_ptr<IEngine> create(EngineType type, ShapeMode mode);

};
