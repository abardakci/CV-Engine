#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvonnxparser;
using namespace nvinfer1;

// Logger sınıfı
class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // WARNING ve üstü mesajları göster
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

bool saveEngine(const std::string& fileName, IHostMemory* engineData)
{
    std::ofstream file(fileName, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open output file for writing: " << fileName << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(engineData->data()), engineData->size());
    file.close();
    return true;
}

int buildEngine(const std::string& onnxModelPath, const std::string& engineFilePath)
{
    Logger logger;

    // Builder oluştur
    IBuilder* builder = createInferBuilder(logger);
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;
        return -1;
    }

    auto flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    // Network oluştur (explicit batch flag ile)
    INetworkDefinition* network = builder->createNetworkV2(flag);    
    if (!network)
    {
        std::cerr << "Failed to create network" << std::endl;
        return -1;
    }

    // ONNX parser oluştur
    IParser* parser = createParser(*network, logger);    
    if (!parser)
    {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return -1;
    }

    parser->parseFromFile(onnxModelPath.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // Builder config oluştur
    IBuilderConfig* config = builder->createBuilderConfig();    
    if (!config)
    {
        std::cerr << "Failed to create builder config" << std::endl;
        return -1;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    saveEngine(engineFilePath, serializedModel);

    return 0;
}

int main()
{
    std::cout << "TensorRT version: "
            << NV_TENSORRT_MAJOR << "."
            << NV_TENSORRT_MINOR << "."
            << NV_TENSORRT_PATCH << std::endl;

    int driver_version = 0;
    int runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    std::cout << "CUDA Driver Version: " << driver_version << std::endl;
    std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;

    std::string onnxPath = "/home/alper/code-repo/vision-engine/scripts/yolov8n.onnx";
    std::string enginePath = "/home/alper/code-repo/vision-engine/scripts/yolov8n.plan";

    return buildEngine(onnxPath, enginePath);
}