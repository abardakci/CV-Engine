#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <array>
#include <format>

#include "engine_interface.hpp"
#include "trt_helpers.hpp"
#include "trt_logger.hpp"
#include "file_io.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"

class TrtEngine : public IEngine
{
public:
    TrtEngine(ShapeMode mode);
    ~TrtEngine() override;
    TrtEngine &operator=(TrtEngine &) = delete;
    void init(const std::string &path) override;
    void set_input(const float *input) override;
    void set_input_dynamic(const float *input, std::vector<int> shape, std::vector<int> output_shape) override;
    bool infer() override;
    float *get_output() override;
    std::vector<int> get_output_dims() const override;
    void apply_dynamic_shape_if_needed(int b, int c, int h, int w, std::vector<int> output_dims);

private:
    ShapeMode mode_;
    bool initialized_ = false;
    cudaStream_t stream_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::shared_ptr<IAllocatorBase> allocator_ = AllocatorFactory().create(backend::Tag::CUDA);
    Tensor input_;
    Tensor output_;
};
