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
    TrtEngine() = default;
    ~TrtEngine() override;
    TrtEngine& operator=(TrtEngine&) = delete;
    void init(const std::string& path) override;
    bool infer(std::array<float*, N_MAX_INPUT> inputs, std::array<float*, N_MAX_OUTPUT> outputs) override;

protected:
    void buildEngine(const std::string& engine_path);
    void setIOTensorNames();
    void setTensorShape(const std::string& tensor_name, const std::vector<int>& default_shape);

private:
    cudaStream_t stream_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;

    int n_input_ = 0;
    int n_output_ = 0;

    std::array<std::string, N_MAX_INPUT> input_names_;
    std::array<std::string, N_MAX_OUTPUT> output_names_;

    std::array<nvinfer1::Dims, N_MAX_INPUT> input_shapes_;
    std::array<nvinfer1::Dims, N_MAX_OUTPUT> output_shapes_;

    std::array<float*, N_MAX_INPUT> d_input_buffers_;
    std::array<float*, N_MAX_OUTPUT> d_output_buffers_;

    std::array<size_t, N_MAX_INPUT> input_sizes_;
    std::array<size_t, N_MAX_OUTPUT> output_sizes_;

};
