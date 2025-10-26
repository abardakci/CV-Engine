#include "kalman_factory.hpp"
#include "kf_wrapper.hpp"   // implementasyon sınıfı header'ı

std::unique_ptr<IKalmanFilter> KalmanFactory::create(KalmanType type)
{
    switch (type)
    {
        case KalmanType::KFWrapper:
            return std::make_unique<KFWrapper>();
        default:
            throw std::invalid_argument("Unknown KalmanType");
    }
}
