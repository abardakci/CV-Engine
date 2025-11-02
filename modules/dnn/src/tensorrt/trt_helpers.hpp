#pragma once

#include <NvInfer.h>
#include "cuda_runtime_api.h"

#include <iostream>
#include <memory>
#include <vector>

#include "file_io.hpp"
#include "logger.hpp"

nvinfer1::IRuntime* create_runtime(Logger &logger);
nvinfer1::ICudaEngine* deserialize_engine(nvinfer1::IRuntime *runtime, const std::vector<char> &plan_binary);
nvinfer1::ICudaEngine* create_engine(nvinfer1::IRuntime* runtime, const std::string &model_path);
nvinfer1::IExecutionContext* create_ctx(nvinfer1::ICudaEngine *engine);

size_t calc_elem_size(nvinfer1::Dims dims);
