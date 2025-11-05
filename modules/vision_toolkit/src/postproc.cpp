#include "postproc.hpp"

float iou(const forge::Box &b1, const forge::Box &b2)
{
    if (b1.xyxy_.x1 > b2.xyxy_.x2 || b1.xyxy_.x2 < b2.xyxy_.x1 ||
        b1.xyxy_.y1 > b2.xyxy_.y2 || b1.xyxy_.y2 < b2.xyxy_.y1)
    {
        return 0.0f;
    }

    int x_left = std::max(b1.xyxy_.x1, b2.xyxy_.x1);
    int y_left = std::max(b1.xyxy_.y1, b2.xyxy_.y1);
    int x_right = std::min(b1.xyxy_.x2, b2.xyxy_.x2);
    int y_right = std::min(b1.xyxy_.y2, b2.xyxy_.y2);

    int inter_width = x_right - x_left;
    int inter_height = y_right - y_left;

    int inter_area = inter_width * inter_height;

    int b1_area = (b1.xyxy_.x2 - b1.xyxy_.x1) * (b1.xyxy_.y2 - b1.xyxy_.y1);
    int b2_area = (b2.xyxy_.x2 - b2.xyxy_.x1) * (b2.xyxy_.y2 - b2.xyxy_.y1);

    int union_area = b1_area + b2_area - inter_area;

    if (union_area == 0)
    {
        return 0.0f;
    }

    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

static bool bbox_comparator(const forge::Box &b1, const forge::Box &b2)
{
    return b1.conf_score_ > b2.conf_score_;
}

void nms(std::vector<forge::Box> &bboxes, float iou_threshold, bool is_sorted)
{
    if (!is_sorted)
        std::sort(bboxes.begin(), bboxes.end(), bbox_comparator);

    std::vector<forge::Box> result_boxes;
    std::vector<bool> suppressed(bboxes.size(), false);

    for (int i = 0; i < bboxes.size(); ++i)
    {
        if (suppressed[i])
            continue;

        result_boxes.push_back(bboxes[i]);

        for (int j = i + 1; j < bboxes.size(); ++j)
        {
            if (suppressed[j])
            {
                continue;
            }

            float current_iou = iou(bboxes[i], bboxes[j]);

            if (current_iou > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }

    bboxes = result_boxes;
}
