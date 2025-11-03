#include "trt_engine.hpp"

using namespace nvinfer1;

Logger gLogger;
TensorFactory tensorFactory;

TrtEngine::~TrtEngine() {}

void TrtEngine::init(const std::string &path)
{
    if (initialized_)
        return;
    initialized_ = true;

    // Create engine & context
    runtime_.reset(create_runtime(gLogger));
    engine_.reset(create_engine(runtime_.get(), path));
    ctx_.reset(create_ctx(engine_.get()));

    // Set input and output tensors
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_ = tensorFactory.create(engine_.get(), allocator_, i);

        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            output_ = tensorFactory.create(engine_.get(), allocator_, i);
    }

    ctx_->setInputTensorAddress(input_.name_.c_str(), input_.buffer_.d_data_);
    ctx_->setOutputTensorAddress(output_.name_.c_str(), output_.buffer_.d_data_);
}

void TrtEngine::set_input(const float *input)
{
    allocator_->memcpy_async(input_.buffer_.h_data_, (void *)input, input_.buffer_.size_ * sizeof(float), backend::CopyMode::HostToHost);
}

float *TrtEngine::get_output()
{
    return output_.buffer_.h_data_;
}

bool TrtEngine::infer()
{
    bool success;

    allocator_->memcpy_async(input_.buffer_.d_data_, input_.buffer_.h_data_, input_.buffer_.size_, backend::CopyMode::HostToDevice);

    success = ctx_->enqueueV3(stream_);
    if (!success)
    {
        std::cout << std::format("[ERROR] enqueueV3 not succeed!");
        return false;
    }

    allocator_->memcpy_async(output_.buffer_.h_data_, output_.buffer_.d_data_, input_.buffer_.size_, backend::CopyMode::DeviceToHost);
    cudaStreamSynchronize(stream_);

    return true;
}
