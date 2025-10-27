#include "engine.hpp"

using namespace nvinfer1;

Logger gLogger;

TrtEngine::~TrtEngine()
{
    for (int i = 0; i < n_input_; i++)
        cudaFree(d_input_buffers_[i]);
    for (int i = 0; i < n_output_; i++)
        cudaFree(d_output_buffers_[i]);

    cudaStreamDestroy(stream_);
}

void TrtEngine::init(const std::string &path)
{
    runtime_.reset(createInferRuntime(gLogger));
    buildEngine(path);

    ctx_.reset(engine_->createExecutionContext());
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)
            ++n_input_;
        else if (mode == TensorIOMode::kOUTPUT)
            ++n_output_;
    }

    if (n_input_ > N_MAX_INPUT || n_output_ > N_MAX_OUTPUT)
        std::cout << std::format("[ERROR] Not enough io buffers!");

    setIOTensorNames();

    // fix
    input_shapes_[0] = nvinfer1::Dims4(1, 3, 640, 640);
    output_shapes_[0] = nvinfer1::Dims2(84, 8400);

    for (int i = 0; i < n_input_; i++)
    {
        size_t tensor_size = calcTensorSize(input_shapes_[i]) * sizeof(float);
        CUDA_CHECK(
            cudaMalloc((void **)&d_input_buffers_[i], tensor_size));

        ctx_->setTensorAddress(input_names_[i].c_str(), d_input_buffers_[i]);
        input_sizes_[i] = tensor_size;
    }

    for (int i = 0; i < n_output_; i++)
    {
        size_t tensor_size = calcTensorSize(output_shapes_[i]) * sizeof(float);
        CUDA_CHECK(
            cudaMalloc((void **)&d_output_buffers_[i], tensor_size));

        ctx_->setTensorAddress(output_names_[i].c_str(), d_output_buffers_[i]);
        output_sizes_[i] = tensor_size;
    }

    CUDA_CHECK(
        cudaStreamCreate(&stream_));
}

void TrtEngine::setIOTensorNames()
{
    int num_io = engine_->getNbIOTensors();
    int input_idx = 0;
    int output_idx = 0;
    for (int i = 0; i < num_io; ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_names_[input_idx] = name;
            ++input_idx;
        } 
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output_names_[output_idx] = name;
            ++output_idx;
        }
    }
}

void TrtEngine::buildEngine(const std::string &model_path)
{
    std::vector<char> model_bin;
    load_binary_file(model_path, model_bin);

    engine_.reset(runtime_->deserializeCudaEngine(model_bin.data(), model_bin.size()));
}

void TrtEngine::setTensorShape(const std::string &tensor_name, const std::vector<int> &default_shape)
{
    nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());

    if (dims.nbDims != static_cast<int>(default_shape.size()))
    {
        std::cerr << "[TensorRT] Shape mismatch for " << tensor_name
                  << ": expected " << dims.nbDims
                  << " dims, got " << default_shape.size() << std::endl;
        return;
    }

    // -1 olanları doldur
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] == -1)
            dims.d[i] = default_shape[i];
    }

    int ret = ctx_->setInputShape(tensor_name.c_str(), dims);
    if (ret < 0)
    {
        std::cerr << "[TensorRT] Failed to set input shape for "
                  << tensor_name << std::endl;
    }
}

bool TrtEngine::infer(std::array<float *, N_MAX_INPUT> inputs, std::array<float *, N_MAX_OUTPUT> outputs)
{
    bool success;
    for (int i = 0; i < n_input_; i++)
    {
        CUDA_CHECK(
            cudaMemcpyAsync((void *)d_input_buffers_[i], (void *)inputs[i], input_sizes_[i], cudaMemcpyHostToDevice, stream_));
    }

    success = ctx_->enqueueV3(stream_);
    if (!success)
    {
        std::cout << std::format("[ERROR] enqueueV3 not succeed!");
        return false;
    }

    for (int i = 0; i < n_output_; i++)
    {
        CUDA_CHECK(
            cudaMemcpyAsync((void *)outputs[i], (void *)d_output_buffers_[i], output_sizes_[i], cudaMemcpyDeviceToHost, stream_));
    }

    CUDA_CHECK(
        cudaStreamSynchronize(stream_));

    return true;
}
