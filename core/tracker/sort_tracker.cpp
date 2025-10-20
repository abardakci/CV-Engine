#include "sort_tracker.hpp"

Tracker::Tracker() {}
Tracker::~Tracker() {}

void Tracker::estimateAllTracks() 
{
    for (auto& track : tracks_)
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
            [age_limit](const Track& t) {
                return t.age_ > age_limit;
            }
        ),
        tracks_.end());
}

void Tracker::addNewTrack(Box& b)
{
    tracks_.emplace_back(std::move(b), ++highest_id_);
}

void Tracker::SORT(std::vector<Box>& detects)
{
    if (detects.size() == 0 || tracks_.size() == 0) 
    {
        // Unmatched detections will be created as new tracks
        for (auto& detection : detects)
        {
            addNewTrack(detection);
        }

        return;
    }

    // Kalman filter prediction for all activate tracks
    estimateAllTracks();

    constexpr float threshold = 50.0f;
    cv::Mat cost_mat = computeCostMatrix(tracks_, detects);

    // Maximum weight matching assignment (hungarian algorithm)
    std::vector<int> assignments = hungarian(cost_mat);
    
    std::vector<bool> used_detections(detects.size(), false);

    for (int i = 0; i < assignments.size(); ++i)
    {
        int label = assignments[i]; // assigned detection of ith track

        if (label != -1)
        {
            if (i < cost_mat.rows && label < cost_mat.cols && cost_mat.at<float>(i, label) <= threshold)
            {
                tracks_[i].kf_->correct(detects[label].xywh_.x, detects[label].xywh_.y);
                used_detections[label] = true;
            }
            else
            {
                if (i < tracks_.size())
                    tracks_[i].age_ += 1;
            }
        }
        else
        { 
            tracks_[i].age_ += 1;
        }
    }

    constexpr int age_threshold = 10;
    removeOldTracks(age_threshold);

    // For unmatched detections, create new tracks 
    for (int i = 0; i < detects.size(); ++i)
    {
        if (!used_detections[i])
        {
            addNewTrack(detects[i]);
        }
    }

}
