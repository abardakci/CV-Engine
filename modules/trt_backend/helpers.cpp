#include "helpers.hpp"

void printOptimizationProfiles(const ICudaEngine* engine)
{
    int n_profiles = engine->getNbOptimizationProfiles();
    int n_io = engine->getNbIOTensors();

    std::cout << "Total optimization profiles: " << n_profiles << std::endl;

    for (int p = 0; p < n_profiles; ++p)
    {
        std::cout << "\n=== Profile " << p << " ===" << std::endl;

        for (int i = 0; i < n_io; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
            if (mode != nvinfer1::TensorIOMode::kINPUT)
                continue;

            std::cout << "Input tensor: " << name << std::endl;

            for (auto sel : {OptProfileSelector::kMIN,
                             OptProfileSelector::kOPT,
                             OptProfileSelector::kMAX})
            {
                Dims d = engine->getProfileShape(name, p, sel);

                const char* label = (sel == OptProfileSelector::kMIN) ? "MIN" :
                                    (sel == OptProfileSelector::kOPT) ? "OPT" : "MAX";

                std::cout << "  " << label << ": [ ";
                for (int j = 0; j < d.nbDims; ++j)
                    std::cout << d.d[j] << (j + 1 < d.nbDims ? ", " : " ");
                std::cout << "]" << std::endl;
            }
        }
    }
}
