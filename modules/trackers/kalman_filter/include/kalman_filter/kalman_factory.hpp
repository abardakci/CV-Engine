#pragma once

#include <memory>
#include <stdexcept>
#include "kalman_interface.hpp"

enum class KalmanType
{
    KFWrapper
};

class KalmanFactory
{
public:
    ~KalmanFactory() = default;
    std::unique_ptr<IKalmanFilter> create(KalmanType type);
};
