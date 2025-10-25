#pragma once

#include <NvInfer.h>
#include "cuda_runtime_api.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <format>

#include "file_io.hpp"
#include "engine_interface.hpp"
#include "logger.hpp"
#include "helpers.hpp"

class TrtEngine : public IEngine
{
public:
    ~TrtEngine() override;
    void initialize(const std::string& path) override;
    int infer(std::array<float*, N_MAX_INPUT> inputs, std::array<float*, N_MAX_OUTPUT> outputs) override;

protected:
    void buildEngine(const std::string& engine_path);
    void setIOTensorNames();
    void setTensorShape(const std::string& tensor_name, const std::vector<int>& default_shape);

private:
    cudaStream_t stream_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* ctx_;

    int n_input_;
    int n_output_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<nvinfer1::Dims> input_shapes_;
    std::vector<nvinfer1::Dims> output_shapes_;
    std::vector<float*> d_input_buffers_;
    std::vector<float*> d_output_buffers_;
    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;

};
