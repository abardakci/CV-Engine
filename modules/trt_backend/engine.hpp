#pragma once

#include <NvInfer.h>
#include "cuda_runtime_api.h"

#include <vector>
#include <string>
#include <array>
#include <format>

#include "engine_interface.hpp"
#include "helpers.hpp"
#include "logger.hpp"
#include "file_io.hpp"
#include "tensor.hpp"

class TrtEngine : public IEngine
{
public:
    TrtEngine() = default;
    ~TrtEngine() override;
    TrtEngine &operator=(TrtEngine &) = delete;
    void init(const std::string &path) override;
    void set_input(float *input) override;
    bool infer() override;
    float *get_output() override;

private:
    cudaStream_t stream_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    Tensor input_;
    Tensor output_;
};
