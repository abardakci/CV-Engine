#include "trt_engine.hpp"

using namespace nvinfer1;

void Logger::log(ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= ILogger::Severity::kWARNING) 
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TrtEngine::TrtEngine(ICudaEngine& engine) : m_ctx(engine.createExecutionContext())
{

}

TrtEngine::~TrtEngine()
{

}

int TrtEngine::infer(const cv::Mat& input, cv::Mat& output) 
{
    int error;

    // Giriş kontrolü
    if (input.empty() || input.type() != CV_8UC3) 
    {
        std::cerr << "Input must be non-empty CV_8UC3 image" << std::endl;
        return -1;
    }

    const ICudaEngine& engine = m_ctx->getEngine();
    
    const int scaleFactor = 2;
    const int tileSize = 128; 
    
    const int in_h = input.rows;
    const int in_w = input.cols;
    const int io_c = 3;
    const int out_h = in_h * scaleFactor;
    const int out_w = in_w * scaleFactor;

    const int wTileNum = std::ceil((float)in_w / tileSize);
    const int hTileNum = std::ceil((float)in_h / tileSize);  
    const int tileNum = wTileNum * hTileNum;
    const size_t inputTileBytes = tileSize * tileSize * io_c * sizeof(float);
    const size_t outputTileBytes = inputTileBytes * scaleFactor * scaleFactor;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    error = m_ctx->setOptimizationProfileAsync(0, stream);
    if (error < 0) 
    {
        std::cerr << "Failed to set optimization profile" << std::endl;
        return -1;
    }

    error = m_ctx->setInputShape("input", Dims4(tileNum, io_c, tileSize, tileSize));
    if (error < 0) 
    {
        std::cerr << "Failed to set input shape" << std::endl;
        return -1;
    }

    // Pinned memory yönetimi
    static float* pinnedInput = nullptr;
    static float* pinnedOutput = nullptr;
    static size_t lastInputBytes = 0, lastOutputBytes = 0;
    
    size_t inputBytes = tileNum * inputTileBytes;
    size_t outputBytes = tileNum * outputTileBytes;

    if (!pinnedInput || inputBytes != lastInputBytes) 
    {
        if (pinnedInput) 
        {
            CUDA_CHECK(cudaFreeHost(pinnedInput));
            pinnedInput = nullptr;
        }
        CUDA_CHECK(cudaHostAlloc((void**)&pinnedInput, inputBytes, cudaHostAllocMapped));
        lastInputBytes = inputBytes;
    }

    if (!pinnedOutput || outputBytes != lastOutputBytes) 
    {
        if (pinnedOutput) 
        {
            CUDA_CHECK(cudaFreeHost(pinnedOutput));
            pinnedOutput = nullptr;
        }

        CUDA_CHECK(cudaHostAlloc((void**)&pinnedOutput, outputBytes, cudaHostAllocMapped));
        lastOutputBytes = outputBytes;
    }

    // Giriş işleme: Normalizasyon ve renk dönüşümü
    // cv::Mat inputFloat;
    // input.convertTo(inputFloat, CV_32FC3, 1.0f/255.0f); // 0-1 normalizasyonu
    // cv::cvtColor(inputFloat, inputFloat, cv::COLOR_BGR2RGB); // BGR -> RGB

    // // CHW düzenine çevir
    // cv::Mat chw_input(io_c, in_h * in_w, CV_32F, pinnedInput);
    // std::vector<cv::Mat> channels(io_c);
    // for (int i = 0; i < io_c; ++i) 
    // {
    //     channels[i] = chw_input.row(i).reshape(1, in_h);
    // }
    // cv::split(inputFloat, channels);

    cv::Mat inputRGB, inputFloat;
    cv::cvtColor(input, inputRGB, cv::COLOR_BGR2RGB);
    input.convertTo(inputFloat, CV_32FC3, 1.0f/255.0f);

    getTiles(inputFloat, pinnedInput, tileSize); // row major, also handles hwc to chw transform

    if (!m_ctx->setTensorAddress("input", pinnedInput) || 
        !m_ctx->setTensorAddress("output", pinnedOutput)) 
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        return -1;
    }
    auto start = timer::now();

    if (!m_ctx->enqueueV3(stream)) 
    {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = timer::now();
    printTime("enqueue time", start, end);

    output.create(out_h, out_w, CV_32FC3);

    for (int t = 0; t < tileNum; ++t)
    {
        float* tileOffset = pinnedOutput + t * outputTileBytes;
        int tileOut_h = tileSize * scaleFactor;
        int tileOut_w = tileSize * scaleFactor;
     
        std::vector<cv::Mat> out_channels(io_c);
        for (int c = 0; c < io_c; ++c) 
        {
            out_channels[c] = cv::Mat(out_h, out_w, CV_32F, tileOffset + c * out_h * out_w).clone();
        }
        
        cv::Mat outputTile;
        cv::merge(out_channels, outputTile);
        
        int roiRow = t / hTileNum;
        int roiCol = t % wTileNum;

        cv::Rect roi(roiCol * out_w, roiRow * out_h, out_w, out_h);
        outputTile.copyTo(output(roi));
    }

    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    output.convertTo(output, CV_8UC3, 255.0f);

    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
