#include "sort_tracker.hpp"
#include "hungarian.hpp"
#include <opencv2/opencv.hpp>

static cv::Mat compute_cost_mat(const std::vector<Track> &tracks, const std::vector<Box> &detects);
KalmanFactory kf_factory;

Tracker::Tracker(KalmanType kalman_type)
{
    kf_type_ = kalman_type;
}

Tracker::~Tracker() {}

void Tracker::estimateAllTracks()
{
    for (auto &track : tracks_)
    {
        track.kf_->predict();
    }
}

void Tracker::removeOldTracks(const int age_limit)
{
    // For unmatched tracks, delete if too old
    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [age_limit](const Track &t)
            {
                return t.age_ > age_limit;
            }),
        tracks_.end());
}

void Tracker::addNewTrack(Box &b)
{
    std::unique_ptr<IKalmanFilter> kf = kf_factory.create(kf_type_);
    tracks_.emplace_back(std::move(b), highest_id_++, std::move(kf));
}

void Tracker::SORT(std::vector<Box> &detects)
{
    if (detects.size() == 0 || tracks_.size() == 0)
    {
        // Unmatched detections will be created as new tracks
        for (auto &detection : detects)
        {
            addNewTrack(detection);
        }

        return;
    }

    // Kalman filter prediction for all activate tracks
    estimateAllTracks();

    constexpr float threshold = 5.0f;
    cv::Mat cost_mat = compute_cost_mat(tracks_, detects);

    // Maximum weight matching assignment (hungarian algorithm)
    std::vector<int> assignments = hungarian(cost_mat);

    std::vector<bool> used_detections(detects.size(), false);

    int size = assignments.size();
    for (int i = 0; i < size; ++i)
    {
        int label = assignments[i]; // assigned detection of ith track

        // matched
        if (label != -1)
        {
            if (i < cost_mat.rows && label < cost_mat.cols && cost_mat.at<float>(i, label) <= threshold)
            {
                tracks_[i].kf_->correct({detects[label].xywh_.x, detects[label].xywh_.y});
                used_detections[label] = true;
                tracks_[i].age_ = 0;
            }
            else
            {
                if (i < tracks_.size())
                    tracks_[i].age_ += 1;
            }
        }
        // unmatched
        else
        {
            tracks_[i].age_ += 1;
        }
    }

    constexpr int age_threshold = 10;
    removeOldTracks(age_threshold);

    // For unmatched detections, create new tracks
    size = detects.size(); 
    for (int i = 0; i < size; ++i)
    {
        if (!used_detections[i])
        {
            addNewTrack(detects[i]);
        }
    }
}

static cv::Mat compute_cost_mat(const std::vector<Track> &tracks, const std::vector<Box> &detects)
{
    int w = detects.size();
    int h = tracks.size();

    cv::Mat cost_mat(cv::Size(w, h), CV_32F);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            float distance = std::hypot(detects[j].xywh_.x - tracks[i].bbox_.xywh_.x,
                                        detects[j].xywh_.y - tracks[i].bbox_.xywh_.y);
            cost_mat.at<float>(i, j) = distance;
        }
    }
    
    return cost_mat;
}
