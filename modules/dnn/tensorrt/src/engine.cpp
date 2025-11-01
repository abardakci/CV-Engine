#include "engine.hpp"

using namespace nvinfer1;

Logger gLogger;
TensorFactory tensorFactory;

TrtEngine::~TrtEngine()
{
    cudaStreamDestroy(stream_);
}

void TrtEngine::init(const std::string &path)
{
    if (initialized_) return;
    initialized_ = true;

    // Create engine & context
    runtime_.reset(create_runtime(gLogger));
    engine_.reset(create_engine(runtime_.get(), path));
    ctx_.reset(create_ctx(engine_.get()));
    
    // Set input and output tensors
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)
            input_ = tensorFactory.create(engine_.get(), i);

        else if (mode == TensorIOMode::kOUTPUT)
            output_ = tensorFactory.create(engine_.get(), i);
    }

    ctx_->setInputTensorAddress(input_.name_.c_str(), input_.d_buffer_);
    ctx_->setOutputTensorAddress(output_.name_.c_str(), output_.d_buffer_);

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

void TrtEngine::set_input(float *input)
{
    input_.h_buffer_ = input;

    CUDA_CHECK(cudaMemcpyAsync(input_.d_buffer_, input_.h_buffer_, input_.elem_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
}

float *TrtEngine::get_output()
{
    return (float *)output_.h_buffer_;
}

bool TrtEngine::infer()
{
    bool success;

    success = ctx_->enqueueV3(stream_);
    if (!success)
    {
        std::cout << std::format("[ERROR] enqueueV3 not succeed!");
        return false;
    }

    CUDA_CHECK(cudaMemcpyAsync(output_.h_buffer_, output_.d_buffer_, output_.elem_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return true;
}
