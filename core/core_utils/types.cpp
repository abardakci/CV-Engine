#include "types.hpp"

// ---------------------------------------
// Box Class Implementation
// ---------------------------------------

Box::Box(float x1, float y1, float x2, float y2, int class_id, float conf_score)
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

// ---------------------------------------
// Track Class Implementation
// ---------------------------------------

Track::Track(Box bbox, int track_id, std::unique_ptr<IKalmanFilter> kf)
    : bbox_(bbox), kf_(std::move(kf)), track_id_(track_id), age_(0)
{
    std::vector<float> current_state{bbox_.xywh_.x, bbox_.xywh_.y};
    kf_->init(current_state);
}

void Track::update(Box &true_bbox)
{
    std::vector<float> measurement{true_bbox.xywh_.x, true_bbox.xywh_.y};
    kf_->correct(measurement);
    bbox_ = std::move(true_bbox);
    age_ = 0;
}
