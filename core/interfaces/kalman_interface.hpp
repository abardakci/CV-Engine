#pragma once

class IKalmanFilter {
public:
    virtual ~IKalmanFilter() = default;
    virtual void init(float x, float y) = 0;
    virtual void correct(float measured_x, float measured_y) = 0;
    virtual void predict() = 0;
    virtual float getX() const = 0;
    virtual float getY() const = 0;
};
