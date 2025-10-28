#include "sort_tracker.hpp"
#include "hungarian.hpp"
#include "postproc.hpp"
#include <opencv2/opencv.hpp>

enum class CostType
{
    EUCLIDEAN,
    IOU
};

static cv::Mat compute_cost_mat(const std::vector<Track> &tracks, const std::vector<Box> &detects, CostType cost_type);
KalmanFactory kf_factory;

Tracker::Tracker(SORTConfig sort_cfg, KalmanType kalman_type)
{
    cfg_ = sort_cfg;
    kf_type_ = kalman_type;
}

Tracker::~Tracker() {}

void Tracker::addNewTrack(Box &b)
{
    std::unique_ptr<IKalmanFilter> kf = kf_factory.create(kf_type_);
    tracks_.emplace_back(std::move(b), highest_id_++, std::move(kf));
}

void Tracker::SORT(std::vector<Box> &detects)
{
    static constexpr CostType cost_type = CostType::IOU;
    const float threshold = cfg_.assignment_threshold;
    const int age_threshold = cfg_.max_age;

    size_t detects_size = detects.size();
    size_t tracks_size = tracks_.size();

    if (tracks_size == 0)
    {
        // If no existing tracks, add all detections as new tracks
        for (auto &det : detects)
        {
            addNewTrack(det);
        }
        return;
    }

    // Kalman filter prediction for all activate tracks
    for (auto &track : tracks_)
    {
        track.kf_->predict();
    }

    // Compute cost matrix (Euclidean or IoU distance)
    cv::Mat cost_mat = compute_cost_mat(tracks_, detects, cost_type); // [tracks x detects]

    // Maximum weight matching assignment (hungarian algorithm)
    std::vector<int> assignments = hungarian(cost_mat.ptr<float>(), cost_mat.rows, cost_mat.cols);
    std::vector<bool> unmatched_tracks(tracks_size, false);
    std::vector<bool> unmatched_detections(detects_size, false);

    // Mar unmatched tracks and detections based on assignment threshold
    for (int i = 0; i < tracks_size; ++i)
    {
        if (assignments[i] >= detects_size || cost_mat.at<float>(i, assignments[i]) > threshold)
        {
            unmatched_tracks[i] = true;
            unmatched_detections[assignments[i]] = true;
        }
    }

    // Update valid tracks with assigned detections
    for (int i = 0; i < tracks_size; ++i)
    {
        if (unmatched_tracks[i])
        {
            tracks_[i].age_ += 1;
            continue;
        }

        int assigned_det = assignments[i]; // assigned detection of ith track
        tracks_[i].kf_->correct({detects[assigned_det].xywh_.x, detects[assigned_det].xywh_.y});
        tracks_[i].bbox_ = detects[assigned_det];
        tracks_[i].age_ = 0;
    }

    // Remove old tracks
    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [age_threshold](const Track &t)
            {
                return t.age_ > age_threshold;
            }),
        tracks_.end());

    // For unmatched detections, create new tracks
    for (int i = 0; i < detects_size; ++i)
    {
        if (unmatched_detections[i])
        {
            addNewTrack(detects[i]);
        }
    }
}

static cv::Mat compute_cost_mat(const std::vector<Track> &tracks, const std::vector<Box> &detects, CostType cost_type)
{
    int w = detects.size();
    int h = tracks.size();

    cv::Mat cost_mat(cv::Size(w, h), CV_32F);

    float *data = cost_mat.ptr<float>();
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            float cost = 0.0f;
            if (cost_type == CostType::EUCLIDEAN)
            {
                float dx = detects[j].xywh_.x - tracks[i].bbox_.xywh_.x;
                float dy = detects[j].xywh_.y - tracks[i].bbox_.xywh_.y;
                cost = std::hypot(dx, dy);
            }
            else if (cost_type == CostType::IOU)
            {
                float iou_score = iou(tracks[i].bbox_, detects[j]);
                cost = 1.0f - iou_score; // düşük IoU → yüksek cost
            }

            data[i * w + j] = cost;
        }
    }

    return cost_mat;
}
