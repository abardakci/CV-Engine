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

    cv::Mat inputRGB, inputFloat;
    cv::cvtColor(input, inputRGB, cv::COLOR_BGR2RGB);
    input.convertTo(inputFloat, CV_32FC3, 1.0f/255.0f);

    std::vector<cudaStream_t> streams(tileNum);
    for (auto stream : streams)
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        error = m_ctx->setOptimizationProfileAsync(0, stream);
        if (error < 0) 
        {
            std::cerr << "Failed to set optimization profile" << std::endl;
            return -1;
        }    
    }

    error = m_ctx->setInputShape("input", Dims4(1, io_c, tileSize, tileSize));
    if (error < 0) 
    {
        std::cerr << "Failed to set input shape" << std::endl;
        return -1;
    }

    // Pinned memory yönetimi
    // static float* pinnedInput = nullptr;
    // static float* pinnedOutput = nullptr;
    // static size_t lastInputBytes = 0, lastOutputBytes = 0;
    
    // size_t inputBytes = in_h * in_w * io_c * sizeof(float);
    // size_t outputBytes = out_h * out_w * io_c * sizeof(float);

    // if (!pinnedInput || inputBytes != lastInputBytes) 
    // {
    //     if (pinnedInput) 
    //     {
    //         CUDA_CHECK(cudaFreeHost(pinnedInput));
    //         pinnedInput = nullptr;
    //     }

    //     CUDA_CHECK(cudaHostAlloc((void**)&pinnedInput, inputBytes, cudaHostAllocMapped));
    //     lastInputBytes = inputBytes;
    // }

    // if (!pinnedOutput || outputBytes != lastOutputBytes) 
    // {
    //     if (pinnedOutput) 
    //     {
    //         CUDA_CHECK(cudaFreeHost(pinnedOutput));
    //         pinnedOutput = nullptr;
    //     }

    //     CUDA_CHECK(cudaHostAlloc((void**)&pinnedOutput, outputBytes, cudaHostAllocMapped));
    //     lastOutputBytes = outputBytes;
    // }

    std::vector<float*> inMems(tileNum);
    std::vector<float*> outMems(tileNum);

    for (int i = 0; i < tileNum; ++i)
    {
        cudaHostAlloc((void**)&inMems[i], inputTileBytes, cudaHostAllocMapped);
        cudaMemset(inMems[i], 0, inputTileBytes);

        cudaHostAlloc((void**)&outMems[i], outputTileBytes, cudaHostAllocMapped);
    }


    // COPY TILES AS CHW TO INMEMS 

    // CHW düzenine çevir
    // cv::Mat chw_input(io_c, in_h * in_w, CV_32F, pinnedInput);
    // std::vector<cv::Mat> channels(io_c);
    // for (int i = 0; i < io_c; ++i) 
    // {
    //     channels[i] = chw_input.row(i).reshape(1, in_h);
    // }
    // cv::split(inputFloat, channels);

    // getTiles(inputFloat, pinnedInput, wTileNum, hTileNum, tileSize, 0); // row major, also handles hwc to chw transform



    auto start = timer::now();

    for (int i = 0; i < streams.size(); ++i)
    {
        if (!m_ctx->setTensorAddress("input", inMems[i]) || 
            !m_ctx->setTensorAddress("output", outMems[i])) 
        {
            std::cerr << "Failed to set tensor addresses" << std::endl;
            return -1;
        }

        if (!m_ctx->enqueueV3(streams[i])) 
        {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }
    }
    
    for (auto stream : streams)
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto end = timer::now();
    printTime("enqueue time", start, end);

    output.create(out_h, out_w, CV_32FC3);

    // for (int t = 0; t < tileNum; ++t)
    // {
    //     int tileOut_h = tileSize * scaleFactor;
    //     int tileOut_w = tileSize * scaleFactor;
    //     float* tileOffset = pinnedOutput + t * tileOut_h * tileOut_w * io_c;
     
    //     std::vector<cv::Mat> out_channels(io_c);
    //     for (int c = 0; c < io_c; ++c) 
    //     {
    //         out_channels[c] = cv::Mat(out_h, out_w, CV_32F, tileOffset + c * tileOut_h * tileOut_w).clone();
    //     }
        
    //     cv::Mat outputTile;
    //     cv::merge(out_channels, outputTile);
        
    //     int roiRow = t / hTileNum;
    //     int roiCol = t % wTileNum;

    //     cv::Rect roi(roiCol * tileOut_w, roiRow * tileOut_h, tileOut_w, tileOut_h);
    //     outputTile.copyTo(output(roi));
    // }

    // cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    // output.convertTo(output, CV_8UC3, 255.0f);

    for (auto stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
