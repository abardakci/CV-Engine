#pragma once

#include <vector>

class IKalmanFilter
{
public:
    virtual ~IKalmanFilter() = default;
    virtual void init(const std::vector<float> &state) = 0;
    virtual void correct(const std::vector<float> &measurement) = 0;
    virtual void predict(float dt = 1.0f) = 0;
    virtual std::vector<float> get_state() const = 0;
};
