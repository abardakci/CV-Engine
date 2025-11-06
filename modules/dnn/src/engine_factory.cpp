#include "engine_factory.hpp"

#include "trt_engine.hpp"

std::unique_ptr<IEngine> EngineFactory::create(EngineType engine_type, ShapeMode shape)
{
    switch (engine_type)
    {
    case EngineType::TensorRT:
        return std::make_unique<TrtEngine>(shape);

    case EngineType::OnnxRuntime:
        throw std::runtime_error("OnnxRuntime engine not implemented yet");
        
    default:
        throw std::invalid_argument("Unknown EngineType");
    }
}
