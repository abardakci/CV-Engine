#include "types.hpp"
#include "../interfaces/kalman_interface.hpp"

// ---------------------------------------
// Box Class Implementation
// ---------------------------------------

Box::Box(int x1, int y1, int x2, int y2, int class_id, float conf_score)
{
    xyxy_.x1 = x1;
    xyxy_.y1 = y1;
    xyxy_.x2 = x2;
    xyxy_.y2 = y2;

    xywh_.x = (x1 + x2) / 2;
    xywh_.y = (y1 + y2) / 2;
    xywh_.w = x2 - x1;
    xywh_.h = y2 - y1;

    class_id_ = class_id;
    conf_score_ = conf_score;
}

Box::~Box() {}

// ---------------------------------------
// Track Class Implementation
// ---------------------------------------

Track::Track(Box bbox, int track_id, std::unique_ptr<IKalmanFilter> kf) : bbox_(bbox), kf_(std::move(kf)), track_id_(track_id)
{
    kf_->init(bbox.xywh_.x, bbox.xywh_.y);
}

Track::~Track() {}

void Track::update(Box& true_box)
{
    kf_->correct(true_box.xywh_.x, true_box.xywh_.y);
    bbox_ = std::move(true_box);
    age_ = 0;
}
