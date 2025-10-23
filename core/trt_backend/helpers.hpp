#include <iostream>
#include <string>
#include <NvInfer.h>

using namespace nvinfer1;

void printOptimizationProfiles(const ICudaEngine *engine);

inline size_t calcTensorSize(const Dims &dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        if (dims.d[i] != 0)
            size *= dims.d[i];

    return size;
}

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
