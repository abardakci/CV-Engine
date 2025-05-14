#include "trt_engine.hpp"

using namespace nvinfer1;

void Logger::log(ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= ILogger::Severity::kWARNING) 
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TrtEngine::TrtEngine(const std::string& engine_path)
{
    std::vector<char> engine_bin;
    loadBinaryFromFile(engine_path, engine_bin);

    std::unique_ptr<IRuntime> runtime(createInferRuntime(m_logger));
    m_engine = runtime->deserializeCudaEngine(engine_bin.data(), engine_bin.size()); 

    m_ctx = m_engine->createExecutionContext();

    m_input_name = m_engine->getIOTensorName(0);
    m_output_name = m_engine->getIOTensorName(1);

    Dims inputDims  = m_ctx->getTensorShape(m_input_name.c_str());
    Dims outputDims = m_ctx->getTensorShape(m_output_name.c_str());

    m_input_size = 1;
    for (int64_t dim : inputDims.d)
    {
        if (dim != 0)
            m_input_size *= dim;  
    }

    m_output_size = 1;
    for (int64_t dim : outputDims.d)
    {
        if (dim != 0)
            m_output_size *= dim; 
    }

    std::cout << "input name "  << m_input_name  << " - input size "  << m_input_size  << "\n";
    std::cout << "output name " << m_output_name << " - output size " << m_output_size << "\n";

}

TrtEngine::~TrtEngine()
{
    delete m_ctx;
    delete m_engine;
}

int TrtEngine::infer(float* input, float* output)
{
    int error = 0;

    // Stream oluştur
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Device memory tahsisi
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, m_input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, m_output_size * sizeof(float)));

    // Host -> Device kopyala
    CUDA_CHECK(cudaMemcpyAsync(d_input, input, m_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Tensor adresleri
    if (!m_ctx->setTensorAddress(m_input_name.c_str(), d_input) || 
        !m_ctx->setTensorAddress(m_output_name.c_str(), d_output)) 
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    // Inference
    if (!m_ctx->enqueueV3(stream)) {
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