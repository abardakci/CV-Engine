#pragma once

#include <memory>
#include "engine.hpp"

enum class EngineType
{
    TensorRT,
    OnnxRuntime
};

class EngineFactory
{
public:
    virtual std::unique_ptr<IEngine> create(EngineType type)
    {
        switch (type)
        {
        case EngineType::TensorRT:
            return std::make_unique<TrtEngine>();
        case EngineType::OnnxRuntime:
            // return std::make_unique<OnnxRuntimeEngine>(); // OnnxRuntimeEngine class, not yet implemented
        default:
            throw std::invalid_argument("Unknown EngineType");
        }
    }
};
