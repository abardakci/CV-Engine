#include "cv_types.hpp"

Box::Box(int x1, int y1, int x2, int y2, int class_id, float conf_score)
{
    xyxy_.x1 = x1;
    xyxy_.y1 = y1;
    xyxy_.x2 = x2;
    xyxy_.y2 = y2;

    xywh_.x = (x1 + x2) / 2;
    xywh_.y = (y1 + y2) / 2;
    xywh_.w = (x2 - x1) / 2;
    xywh_.h = (y2 - y1) / 2;

    class_id_ = class_id;
    conf_score_ = conf_score;
}

Box::~Box() {}

void Box::print()
{
    std::cout << "x1: "   << xyxy_.x1
              << ", y1: " << xyxy_.y1
              << ", x2: " << xyxy_.x2
              << ", y2: " << xyxy_.y2 << "\n"
              << "id: "   << class_id_ << "\n"
              << "score: " << conf_score_ << "\n";
}

static void initKalmanFilter(cv::KalmanFilter& kf, int x, int y)
{
    // 4 state: x, y, vx, vy
    // 2 measurement: x, y
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0, 
        0, 1, 0, 1, 
        0, 0, 1, 0, 
        0, 0, 0, 1);

    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    kf.statePost = (cv::Mat_<float>(4, 1) << x, y, 0, 0);
}

Track::Track(Box bbox, int track_id) : bbox_(bbox), kf_(4, 2), track_id_(track_id)
{
    initKalmanFilter(kf_, bbox.xywh_.x, bbox.xywh_.y);
}

Track::~Track() {}

int Track::get_xhat() const
{
    return static_cast<int>(kf_.statePost.at<float>(0));
}

int Track::get_yhat() const
{
    return static_cast<int>(kf_.statePost.at<float>(1));
}

void Track::correction(const Box& true_box)
{
    cv::Mat measurement_mat = (cv::Mat_<float>(2, 1) <<
        true_box.xywh_.x,
        true_box.xywh_.y);

    kf_.correct(measurement_mat);

    bbox_ = std::move(true_box);
    age_ = 0;
}
