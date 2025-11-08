#include "engine_factory.hpp"

#include "trt_engine.hpp"

std::unique_ptr<IEngine> EngineFactory::create(EngineType engine_type, ShapeMode shape)
{
    switch (engine_type)
    {
    case EngineType::TensorRT:
    {
        auto f = std::make_unique<TrtEngine>();
            
        if (shape == ShapeMode::Static)
            return f;
        else if (shape == ShapeMode::Dynamic)
            return std::make_unique<TrtEngineDynamic>(std::move(f));
    }
    case EngineType::OnnxRuntime:
        throw std::runtime_error("OnnxRuntime engine not implemented yet");
        
    default:
        throw std::invalid_argument("Unknown EngineType");
    }
}
