#pragma once

#include <memory>
#include <stdexcept>
#include "kalman_interface.hpp"
#include "kf_wrapper.hpp"   // implementasyon sınıfı header'ı

enum class KalmanType { KFWrapper };

class KalmanFactory {
public:
    static inline std::unique_ptr<IKalmanFilter> create(KalmanType type)
    {
        switch (type)
        {
            case KalmanType::KFWrapper:
                return std::make_unique<KFWrapper>();
            default:
                throw std::invalid_argument("Unknown KalmanType");
        }
    }
};
