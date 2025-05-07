#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
#include "cuda_runtime.h"

// Hata kontrol makrosu
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}