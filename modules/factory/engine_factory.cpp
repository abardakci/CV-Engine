#include "engine_factory.hpp"
#include "engine.hpp"

std::unique_ptr<IEngine> EngineFactory::create(EngineType engine_type)
{
    switch (engine_type)
    {
    case EngineType::TensorRT:
        return std::make_unique<TrtEngine>();

    case EngineType::OnnxRuntime:
        throw std::runtime_error("OnnxRuntime engine not implemented yet");
    default:
        throw std::invalid_argument("Unknown EngineType");
    }
}
