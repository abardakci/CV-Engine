#include "types.hpp"

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
