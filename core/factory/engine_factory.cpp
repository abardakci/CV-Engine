#include "engine_factory.hpp"
#include "engine.hpp"

std::unique_ptr<IEngine> EngineFactory::create(EngineType type)
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
