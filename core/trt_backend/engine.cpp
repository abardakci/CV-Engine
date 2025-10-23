#include "engine.hpp"

using namespace nvinfer1;

TrtEngine::~TrtEngine()
{
    delete ctx_;
    delete engine_;
    delete runtime_;
}

void TrtEngine::initialize(const std::string &path)
{
    runtime_ = createInferRuntime(gLogger);
    buildEngine(path);
    
    ctx_ = engine_->createExecutionContext();
    setIOTensorNames();
    CUDA_CHECK(
        cudaStreamCreate(&stream_)
    );
}

void TrtEngine::setIOTensorNames()
{
    int num_io = engine_->getNbIOTensors();
    for (int i = 0; i < num_io; ++i)
    {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_names_.push_back(name);
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            output_names_.push_back(name);
    }
}

void TrtEngine::buildEngine(const std::string &model_path)
{
    std::vector<char> model_bin;
    loadBinaryFromFile(model_path, model_bin);

    engine_ = runtime_->deserializeCudaEngine(model_bin.data(), model_bin.size());
}

void TrtEngine::setTensorShape(const std::string& tensor_name, const std::vector<int>& default_shape)
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

int TrtEngine::infer(std::vector<float*> inputs, std::vector<float*> outputs)
{
    int n_input = input_names_.size();
    int n_output = output_names_.size();
    std::vector<float*> d_input_mem(n_input);
    std::vector<float*> d_output_mem(n_output);
    
    bool success = true;
    int idx = 0;
    for (float* mem : d_input_mem)
    {
        size_t tensor_size = calcTensorSize(input_shapes_[idx]) * sizeof(float);
        CUDA_CHECK(
            cudaMalloc(&mem, tensor_size)
        );
        
        CUDA_CHECK(
            cudaMemcpyAsync(mem, inputs[idx], tensor_size, cudaMemcpyHostToDevice, stream_)
        );

        success &= ctx_->setTensorAddress(input_names_[idx].c_str(), mem);
    }

    idx = 0;
    for (float* mem : d_output_mem)
    {
        size_t tensor_size = calcTensorSize(output_shapes_[idx]) * sizeof(float);
        CUDA_CHECK(
            cudaMalloc(&mem, tensor_size)
        );

        success &= ctx_->setTensorAddress(output_names_[idx].c_str(), mem);
    }

    if (!success)
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        for (auto mem : d_input_mem)
            cudaFree(mem);

        for (auto mem : d_output_mem)
            cudaFree(mem);

        return -1;
    }

    success = ctx_->enqueueV3(stream_);
    if (!success)
    {
        std::cerr << "Failed to enqueueV3" << std::endl;
        for (auto mem : d_input_mem)
            cudaFree(mem);

        for (auto mem : d_output_mem)
            cudaFree(mem);
            
        return -1;
    }

    idx = 0;
    for (float* mem : outputs)
    {
        size_t tensor_size = calcTensorSize(output_shapes_[idx]) * sizeof(float);
        CUDA_CHECK(
            cudaMemcpyAsync(mem, d_output_mem[idx], tensor_size, cudaMemcpyDeviceToHost, stream_)
        );
    }

    CUDA_CHECK(
        cudaStreamSynchronize(stream_)
    );

    for (auto mem : d_input_mem)
        cudaFree(mem);

    for (auto mem : d_output_mem)
        cudaFree(mem);

    return 0;
}
