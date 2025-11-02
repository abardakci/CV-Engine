#pragma once

#include "kalman_interface.hpp"
#include <opencv2/opencv.hpp>

class KFWrapper : public IKalmanFilter
{
public:
    KFWrapper();
    ~KFWrapper() = default;
    void init(const std::vector<float> &state) override;
    void correct(const std::vector<float> &measurement) override;
    void predict(float dt = 1.0f) override;
    std::vector<float> get_state() const override;

private:
    cv::KalmanFilter kf_;
};
