#pragma once

#include "kalman_interface.hpp"
#include <opencv2/opencv.hpp>

class OpenCVKalman : public IKalmanFilter {
public:
    OpenCVKalman();
    void init(float x, float y) override;
    void correct(float measured_x, float measured_y) override;
    void predict() override;
    float getX() const override;
    float getY() const override;

private:
    cv::KalmanFilter kf_;
};
