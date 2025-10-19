#include "opencv_kalman.hpp"

OpenCVKalman::OpenCVKalman() : kf_(4, 2) {}

void OpenCVKalman::init(float x, float y) {
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
    kf_.statePost = (cv::Mat_<float>(4, 1) << x, y, 0, 0);
}

void OpenCVKalman::correct(float measured_x, float measured_y) {
    cv::Mat m = (cv::Mat_<float>(2, 1) << measured_x, measured_y);
    kf_.correct(m);
}

void OpenCVKalman::predict() {
    kf_.predict();
}

float OpenCVKalman::getX() const {
    return kf_.statePost.at<float>(0);
}

float OpenCVKalman::getY() const {
    return kf_.statePost.at<float>(1);
}
