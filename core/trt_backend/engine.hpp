#pragma once

#include <NvInfer.h>
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>

#include "file_io.hpp"
#include "engine_interface.hpp"
#include "logger.hpp"
#include "helpers.hpp"

class TrtEngine : public IEngine
{
public:
    ~TrtEngine() override;
    void initialize(const std::string& path) override;
    int infer(std::vector<float*> inputs, std::vector<float*> outputs) override;

protected:
    void buildEngine(const std::string& engine_path);
    void setIOTensorNames();
    void setTensorShape(const std::string& tensor_name, const std::vector<int>& default_shape);

private:
    cudaStream_t stream_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* ctx_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<nvinfer1::Dims> input_shapes_;
    std::vector<nvinfer1::Dims> output_shapes_;

};
