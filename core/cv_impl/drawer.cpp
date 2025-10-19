#include "drawer.hpp"

void drawBox(cv::Mat &target, const Box &box)
{
    // Renk ve font ayarları
    const cv::Scalar color(0, 120, 0); // Yeşil (BGR formatında)
    const int thickness = 1;
    const float font_scale = 0.3;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;

    // Dikdörtgen çiz
    cv::Point pt1(box.xyxy_.x1, box.xyxy_.y1);
    cv::Point pt2(box.xyxy_.x2, box.xyxy_.y2);
    cv::rectangle(target, pt1, pt2, color, thickness);

    // Etiket metni (class_id + confidence)
    std::string label = "Class " + std::to_string(box.class_id_) +
                        " (" + std::to_string(box.conf_score_).substr(0, 4) + ")";

    // Etiket için arka plan dikdörtgeni
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, 1, &baseline);
    cv::Point text_org(box.xyxy_.x1, box.xyxy_.y1 - 5); // Üstte küçük boşluk bırak

    // Arka planı çiz
    cv::rectangle(
        target,
        cv::Rect(text_org.x, text_org.y - text_size.height,
                 text_size.width, text_size.height + baseline),
        color,
        cv::FILLED);

    // Metni yaz
    cv::putText(
        target,
        label,
        text_org,
        font_face,
        font_scale,
        cv::Scalar(0, 0, 0), // Siyah metin
        1,
        cv::LINE_AA);
}

void drawBoxes(cv::Mat &target, const std::vector<Box> &boxes)
{
    for (auto box : boxes)
    {
        drawBox(target, box);
    }
}

// void drawTrack(cv::Mat &target, Track &track)
// {
//     const cv::Scalar color(150, 0, 0); // Yeşil (BGR formatında)
//     const int thickness = 1;
//     const float font_scale = 0.3;
//     const int font_face = cv::FONT_HERSHEY_SIMPLEX;

//     // Dikdörtgen çiz
//     cv::Point pt1(track.bbox_.xyxy_.x1, track.bbox_.xyxy_.y1);
//     cv::Point pt2(track.bbox_.xyxy_.x2, track.bbox_.xyxy_.y2);
//     cv::rectangle(target, pt1, pt2, color, thickness);

//     std::string label = "Track id " + std::to_string(track.track_id_) +
//                         " (" + std::to_string(track.bbox_.conf_score_).substr(0, 4) + ")";

//     int baseline = 0;
//     cv::Size text_size = cv::getTextSize(label, font_face, font_scale, 1, &baseline);
//     cv::Point text_org(track.bbox_.xyxy_.x1, track.bbox_.xyxy_.y1 - 5); // Üstte küçük boşluk bırak

//     cv::rectangle(
//         target,
//         cv::Rect(text_org.x, text_org.y - text_size.height,
//                  text_size.width, text_size.height + baseline),
//         color,
//         cv::FILLED);

//     cv::putText(
//         target,
//         label,
//         text_org,
//         font_face,
//         font_scale,
//         cv::Scalar(255, 255, 255),
//         1,
//         cv::LINE_AA);
// }

