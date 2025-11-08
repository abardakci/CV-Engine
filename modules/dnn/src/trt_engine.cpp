#include "trt_engine.hpp"

using namespace nvinfer1;

Logger gLogger;
TensorFactory tensorFactory;

TrtEngine::~TrtEngine() { cudaStreamDestroy(stream_); }

void TrtEngine::init(const std::string &path)
{
    if (initialized_)
        return;
    initialized_ = true;
    cudaStreamCreate(&stream_);

    // Create engine & context
    runtime_.reset(create_runtime(gLogger));
    engine_.reset(create_engine(runtime_.get(), path));
    ctx_.reset(create_ctx(engine_.get()));

    // Set input and output tensors
    int input_idx = 0;
    int output_idx = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_ = tensorFactory.create(engine_.get(), backend::Tag::CUDA, i);

        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            output_ = tensorFactory.create(engine_.get(), backend::Tag::CUDA, i);
    }
    
    bind_tensors();
}

bool TrtEngine::bind_tensors()
{
    bool success = true;
    success = ctx_->setInputTensorAddress(input_.name_.c_str(), input_.buffer_.d_data_);
    success = ctx_->setOutputTensorAddress(output_.name_.c_str(), output_.buffer_.d_data_);
    return success;
}

void TrtEngine::set_input(const float *input)
{
    allocator_->memcpy_async(input_.buffer_.h_data_, (void *)input, input_.buffer_.size_, backend::CopyMode::HostToHost, stream_);
}

float *TrtEngine::get_output()
{
    return output_.buffer_.h_data_;
}

bool TrtEngine::infer()
{
    bool success;

    allocator_->memcpy_async(input_.buffer_.d_data_, input_.buffer_.h_data_, input_.buffer_.size_, backend::CopyMode::HostToDevice, stream_);
    success = ctx_->enqueueV3(stream_);
    if (!success)
    {
        std::cout << std::format("[ERROR] enqueueV3 not succeed!");
        return false;
    }

    allocator_->memcpy_async(output_.buffer_.h_data_, output_.buffer_.d_data_, output_.buffer_.size_, backend::CopyMode::DeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return true;
}
