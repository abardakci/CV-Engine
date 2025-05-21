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

Box::~Box()
{

}

void Box::print()
{
    std::cout << "x1: " <<   xyxy_.x1;
    std::cout << ", y1: " << xyxy_.y1;
    std::cout << ", x2: " << xyxy_.x2;
    std::cout << ", y2: " << xyxy_.y2 << "\n";
    std::cout << "class id: " << class_id_ << "\n";
    std::cout << "conf score: " << conf_score_ << "\n";
}

static void initKalmanFilter(cv::KalmanFilter& kf)
{
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0, 
        0, 1, 0, 1, 
        0, 0, 1, 0, 
        0, 0, 0, 1);

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    randn(kf.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));  // ilk tahmin
}

Track::Track(Box b) : bbox_(b)
{
    initKalmanFilter(kf_);
}

Track::~Track()
{

}

int Track::get_xhat() const
{
    return -1;
}

int Track::get_yhat() const
{
    return -1;
}

void Track::update()
{
    kf_.predict();
}

void Track::correction(const Box& b)
{
    std::vector<int> measurement_vec = {b.xywh_.x, b.xywh_.y, b.xywh_.w, b.xywh_.h};  
    cv::Mat measurement_mat = cv::Mat(measurement_vec);

    kf_.correct(measurement_mat);
}
