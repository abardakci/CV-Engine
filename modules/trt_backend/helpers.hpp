#pragma once 

#include <NvInfer.h>
#include "cuda_runtime_api.h"
#include <iostream>

using namespace nvinfer1;

void printOptimizationProfiles(const ICudaEngine *engine);

// Cuda error check macro
#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }
