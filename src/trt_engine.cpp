#include "trt_engine.hpp"

using namespace nvinfer1;

void Logger::log(ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= ILogger::Severity::kWARNING) 
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

Logger logger;

static size_t calc_tensor_size(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] != 0)
            size *= dims.d[i];
    }

    return size;
}

static ICudaEngine* build_engine(IRuntime* runtime, const std::string& model_path)
{
    std::vector<char> model_bin;
    loadBinaryFromFile(model_path, model_bin);

    return runtime->deserializeCudaEngine(model_bin.data(), model_bin.size()); 
}

TrtEngine::TrtEngine(const std::string& model_path)
{    
    m_runtime = createInferRuntime(logger);
    
    m_engine = build_engine(m_runtime, model_path);

    m_ctx = m_engine->createExecutionContext();

    m_input_name = m_engine->getIOTensorName(0);
    m_output_name = m_engine->getIOTensorName(1);

    Dims input_dims  = m_ctx->getTensorShape(m_input_name.c_str());
    Dims output_dims = m_ctx->getTensorShape(m_output_name.c_str());

    m_input_size = calc_tensor_size(input_dims);
    m_output_size = calc_tensor_size(output_dims);
}

TrtEngine::~TrtEngine()
{
    delete m_ctx;
    delete m_engine;
    delete m_runtime;
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
    CUDA_CHECK(cudaMalloc(&d_input, m_input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, m_output_size * sizeof(float)));

    // Host -> Device kopyala
    CUDA_CHECK(cudaMemcpyAsync(d_input, input, m_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Tensor adresleri
    success  = m_ctx->setTensorAddress(m_input_name.c_str(), d_input);
    success  = m_ctx->setTensorAddress(m_output_name.c_str(), d_output);

    if (!success) 
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    // Inference
    success = m_ctx->enqueueV3(stream);
    if (!success) 
    {
        std::cerr << "Inference failed" << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    // Device -> Host kopyala
    CUDA_CHECK(cudaMemcpyAsync(output, d_output, m_output_size * sizeof(float),
               cudaMemcpyDeviceToHost, stream));

    // Senkronizasyon
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Temizlik
    cudaFree(d_input); 
    cudaFree(d_output);
    return 0;
}