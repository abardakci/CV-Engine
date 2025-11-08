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

inline nvinfer1::Dims to_trt_dims(const tensor::Dims &mydims, int nbDims)
{
    nvinfer1::Dims out;
    out.nbDims = nbDims;
    for (int i = 0; i < nbDims; ++i)
        out.d[i] = static_cast<int>(mydims.d[i]);
    return out;
}

class TrtEngine : public IEngine
{
    friend class TrtEngineDynamic;

public:
    ~TrtEngine() override;
    TrtEngine &operator=(TrtEngine &) = delete;
    void init(const std::string &path) override;
    bool bind_tensors() override;
    void set_input(const float *input) override;
    bool infer() override;
    float *get_output() override;

private:
    bool initialized_ = false;
    cudaStream_t stream_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::shared_ptr<IAllocatorBase> allocator_ = AllocatorFactory::create(backend::Tag::CUDA);
    Tensor input_;
    Tensor output_;
};

class TrtEngineDynamic : public IEngine
{
public:
    explicit TrtEngineDynamic(std::unique_ptr<TrtEngine> base)
        : base_(std::move(base)) {}

    void init(const std::string &path) override
    {
        base_->init(path);
        base_->ctx_->setOptimizationProfileAsync(0, base_->stream_);
        base_->ctx_->getTensorShape(base_->output_.name_.c_str());
    }

    bool bind_tensors() override
    {
        return base_->bind_tensors();
    }

    void set_input(const float *input) override
    {
        base_->set_input(input);
    }

    bool infer() override
    {
        return base_->infer();
    }

    float *get_output() override
    {
        return base_->get_output();
    }

    void set_input_shape(int idx, const tensor::Dims &shape) override
    {
        base_->input_.set_shape(shape);

        auto _nv_input_dims = to_trt_dims(shape, 4);
        if (!base_->ctx_->setInputShape(base_->input_.name_.c_str(), _nv_input_dims))
        {
            std::cerr << "ERROR: input shape is out of profile range!\n";
        }
        
        tensor::Dims out_shape;
        const char *inputs[] = {base_->input_.name_.c_str()};
        base_->ctx_->inferShapes(1, inputs);

        auto dims = base_->ctx_->getTensorShape(base_->output_.name_.c_str());
        for (int i = 0; i < 4; ++i) // fix 4 to num of dims
        {
            out_shape.d[i] = dims.d[i];
        }

        base_->output_.set_shape(out_shape);
        base_->bind_tensors();
    }

private:
    std::unique_ptr<TrtEngine> base_; // kompozisyon
};
