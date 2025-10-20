#include "trt_engine.hpp"

using namespace nvinfer1;

void Logger::log(ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= ILogger::Severity::kWARNING) 
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

Logger gLogger;

static size_t tensorSize(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] != 0)
            size *= dims.d[i];
    }

    return size;
}

static ICudaEngine* buildEngine(IRuntime* runtime, const std::string& model_path)
{
    std::vector<char> model_bin;
    loadBinaryFromFile(model_path, model_bin);

    return runtime->deserializeCudaEngine(model_bin.data(), model_bin.size()); 
}

TrtEngine::~TrtEngine()
{
    delete ctx_;
    delete engine_;
    delete runtime_;
}

void TrtEngine::initialize(const std::string &path)
{
    runtime_ = createInferRuntime(gLogger);
    
    engine_ = buildEngine(runtime_, path);

    ctx_ = engine_->createExecutionContext();

    input_name_  = engine_->getIOTensorName(0);
    output_name_ = engine_->getIOTensorName(1);

    input_dims_  = ctx_->getTensorShape(input_name_.c_str());
    output_dims_ = ctx_->getTensorShape(output_name_.c_str());

    input_size_  = tensorSize(input_dims_);
    output_size_ = tensorSize(output_dims_);
}

int TrtEngine::infer(float* input, float* output)
{
    int success = 0;

    // Stream oluştur
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Device memory tahsisi
    float* d_input; 
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size_ * sizeof(float)));

    // Host -> Device kopyala
    CUDA_CHECK(cudaMemcpyAsync(d_input, input, input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Tensor adresleri
    success  = ctx_->setTensorAddress(input_name_.c_str(), d_input);
    success  = ctx_->setTensorAddress(output_name_.c_str(), d_output);

    if (!success) 
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    // Inference
    success = ctx_->enqueueV3(stream);
    if (!success) 
    {
        std::cerr << "Inference failed" << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    // Device -> Host kopyala
    CUDA_CHECK(cudaMemcpyAsync(output, d_output, output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Senkronizasyon
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Temizlik
    cudaFree(d_input); 
    cudaFree(d_output);
    return 0;
}

size_t TrtEngine::input_size() const
{
    return input_size_;
}

size_t TrtEngine::output_size() const
{
    return output_size_;
}
