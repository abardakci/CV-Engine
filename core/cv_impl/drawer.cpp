#include "drawer.hpp"

void drawBox(cv::Mat &target, const Box &box, const std::string& text)
{
    // Renk ve font
    const cv::Scalar color(0, 120, 0);
    const int thickness = 1;
    const float font_scale = 0.3;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;

    // Obj bbox
    cv::Point pt1(box.xyxy_.x1, box.xyxy_.y1);
    cv::Point pt2(box.xyxy_.x2, box.xyxy_.y2);
    cv::rectangle(target, pt1, pt2, color, thickness);

    // Text için rect
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, 1, &baseline);
    cv::Point text_org(box.xyxy_.x1, box.xyxy_.y1 - 5); // Üstte küçük boşluk bırak

    // Text arka plan
    cv::rectangle(
        target,
        cv::Rect(text_org.x, text_org.y - text_size.height,
                 text_size.width, text_size.height + baseline),
        color,
        cv::FILLED);

    // Metni yaz
    cv::putText(
        target,
        text,
        text_org,
        font_face,
        font_scale,
        cv::Scalar(0, 0, 0), // Siyah metin
        1,
        cv::LINE_AA);
}
