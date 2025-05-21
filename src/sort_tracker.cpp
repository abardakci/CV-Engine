#include "sort_tracker.hpp"

static cv::Mat computeCostMatrix(const std::vector<Track>& tracks, const std::vector<Box>& detects)
{
    cv::Mat cost_mat(cv::Size(tracks.size(), detects.size()), CV_32F);
    for (int i = 0; i < tracks.size(); ++i)
    {
        for (int j = 0; j < detects.size(); ++j)
        {
            cost_mat.at<float>(i, j) = euclidianDistance(detects[j].xywh_.x, detects[j].xywh_.y, tracks[i].get_xhat(), tracks[i].get_yhat());
        }
    }

    return cost_mat;
}

Tracker::Tracker()
{

}

Tracker::~Tracker()
{

}

void Tracker::SORT(std::vector<Box>& detects)
{
// Input: detections (bounding boxes) for current frame
// State: active_tracks ← list of existing Kalman filters

// 1. Predict next position of all active_tracks using Kalman Filter
    for (auto track : tracks_)
    {   
        track.update();
    }

// 2. Compute cost_matrix between predicted tracks and current detections
//     → cost = 1 - IoU(predicted_box, detection_box)   # Or Euclidean distance
    cv::Mat cost_mat = computeCostMatrix(tracks_, detects); // ith row, jth col = cost between ith track and jth detection

    std::vector<int> matchings(tracks_.size());
    std::vector<bool> is_available(detects.size());
 
    const float threshold = 10;
// 3. Apply Hungarian Algorithm to find optimal matches between tracks and detections
    for (int i = 0; i < tracks_.size(); ++i)
    {
        float temp_min = FLT_MIN;
        int min_index = -1;
        
        for (int j = 0; j < detects.size(); ++j)
        {
            if (temp_min < cost_mat.at<float>(i, j) && is_available[j])
            {
                temp_min = cost_mat.at<float>(i, j);
                min_index = j;        
            }
        }
        
        if (min_index != -1 && cost_mat.at<float>(i, min_index) < threshold)
        {
            assert(is_available[min_index]);
            is_available[min_index] = false;
            matchings[i] = min_index;
        }
    }

// 4. For each matched pair (track_i, detection_j):
//     a. Update track_i using detection_j (Kalman filter correction)
//     b. Mark track_i as updated
    for (int i = 0; i < tracks_.size(); ++i)
    {
        if (matchings[i] != -1)
        {
            tracks_[i].correction(detects[matchings[i]]);
        }
        else
        {
            tracks_[i].age_ = 1;
        }   
    }

// 5. For unmatched tracks:
//     a. Increase "age" or "missed" count
//     b. If missed too many times, delete the track
    tracks_.erase(
        std::remove_if(
            tracks_.begin(), 
            tracks_.end(), 
            [](const Track& t) { 
                return t.age_ > 10; 
            }
        ), 
        tracks_.end());

// 6. For unmatched detections:
//     a. Initialize new Kalman filters (new tracks)    

    for (int i = 0; i < detects.size(); ++i)
    {
        if (is_available[i])
        {
            tracks_.emplace_back(std::move(detects[i]));
        }
    }

// 7. Return updated list of tracks
    return;
}
