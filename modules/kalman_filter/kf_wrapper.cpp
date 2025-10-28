#include "kf_wrapper.hpp"

KFWrapper::KFWrapper() : kf_(4, 2) {}

void KFWrapper::init(const std::vector<float> &state)
{
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                     0, 1, 0, 1,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1);

    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                                                      0, 1, 0, 0);

    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

    kf_.statePost = (cv::Mat_<float>(4, 1) << state[0], state[1], 0, 0);
}

void KFWrapper::correct(const std::vector<float> &measurement)
{
    cv::Mat m = (cv::Mat_<float>(2, 1) << measurement[0], measurement[1]);
    kf_.correct(m);
}

void KFWrapper::predict(float dt)
{
    kf_.predict();
}

std::vector<float> KFWrapper::get_state() const
{
    float x = kf_.statePost.ptr<float>()[0];
    float y = kf_.statePost.ptr<float>()[1];
    std::vector<float> state {x, y};
    return state;
}
