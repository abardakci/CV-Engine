#include "trt_helpers.hpp"

nvinfer1::IRuntime* create_runtime(Logger &logger)
{
    return nvinfer1::createInferRuntime(logger);
}

nvinfer1::ICudaEngine* deserialize_engine(nvinfer1::IRuntime *runtime, const std::vector<char> &plan_binary)
{
    return runtime->deserializeCudaEngine(plan_binary.data(), plan_binary.size());
}

nvinfer1::ICudaEngine* create_engine(nvinfer1::IRuntime* runtime, const std::string &model_path)
{
    auto file = load_binary_file(model_path);
    nvinfer1::ICudaEngine* engine = deserialize_engine(runtime, file);
    return engine;
}

nvinfer1::IExecutionContext* create_ctx(nvinfer1::ICudaEngine *engine)
{
    return engine->createExecutionContext();
}
