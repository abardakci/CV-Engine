#pragma once

#include <string_view>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <type_traits>

template <typename T>
concept TensorDtype =
    std::same_as<T, float> || std::same_as<T, __half>;

template <TensorDtype T>
class Tensor
{
public:
    Tensor() = default;
    ~Tensor() = default;

private:
    std::string name;
    nvinfer1::Dims dims{};
    bool has_dynamic_shape = false;
    size_t size = 0;
    T *h_buffer = nullptr;
    T *d_buffer = nullptr;
};

inline size_t calc_elem_size(nvinfer1::Dims dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        const int dim = dims.d[i];
        if (dim == 0)
            continue;
        if (dim == -1)
            return -1;

        size *= dims.d[i];
    }

    return size;
}
