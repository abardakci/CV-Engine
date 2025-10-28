#pragma once

#include <memory>
#include "kalman_interface.hpp"

// ---------------------------------------
// Structs -------------------------------
// ---------------------------------------

typedef struct
{
    float x;
    float y;
    float w;
    float h;
} xywh_t;

typedef struct
{
    float x1;
    float y1;
    float x2;
    float y2;
} xyxy_t;

typedef struct
{
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels;
} ImageWrapper;

// ---------------------------------------
// Classes -------------------------------
// ---------------------------------------

class Box
{
public:
    Box(float x1, float y1, float x2, float y2, int class_id, float conf_score);
    ~Box() = default;

public:
    xyxy_t xyxy_;
    xywh_t xywh_;
    int class_id_;
    float conf_score_;
};

class IKalmanFilter; // Forward declaration
class Track
{
public: // constructors and operators
    Track(Box b, int track_id, std::unique_ptr<IKalmanFilter> filter);
    ~Track() = default;
    Track(const Track &) = delete;
    Track &operator=(const Track &) = delete;

    Track(Track &&other) noexcept
        : bbox_(other.bbox_), kf_(std::move(other.kf_)), track_id_(other.track_id_), age_(other.age_) {}

    Track &operator=(Track &&other) noexcept
    {
        if (this != &other)
        {
            this->kf_ = std::move(other.kf_);
            this->track_id_ = other.track_id_;
            this->bbox_ = other.bbox_;
            this->age_ = other.age_;
        }

        return *this;
    }

public: // functions
    void update(Box &b);

public: // variables
    Box bbox_;
    std::unique_ptr<IKalmanFilter> kf_;
    long track_id_;
    int age_ = 0;
};
