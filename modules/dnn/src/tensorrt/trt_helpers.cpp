#include "helpers.hpp"

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

size_t calc_elem_size(nvinfer1::Dims dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        const int dim = dims.d[i];
        if (dim == 0)
            continue;
        if (dim == -1)
            return -1;

        size *= dims.d[i];
    }

    return size;
}
