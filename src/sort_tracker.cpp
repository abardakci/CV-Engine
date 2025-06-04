#include "sort_tracker.hpp" 

// Hungarian Algorithm
std::vector<int> hungarian(cv::Mat& cost_matrix) 
{
    const int rows = cost_matrix.rows;
    const int cols = cost_matrix.cols;
    constexpr float INF = std::numeric_limits<float>::max();

    // Padding to obtain square matrix 
    const int n = std::max(rows, cols);
    cv::Mat cost_padded(cv::Size(n, n), cost_matrix.type(), 1e6);
    cv::Rect roi = cv::Rect(0, 0, cols, rows);
    cost_matrix.copyTo(cost_padded(roi));

    std::vector<int> assignment(n, -1);
    
    // Dual variable initialization
    std::vector<float> u(n, 0), v(n, 0);
    std::vector<int> match_v(n, -1);

    for (int i = 0; i < n; ++i) 
    {
        std::vector<float> min_slack(n, INF);
        std::vector<int> prev(n, -1);
        std::vector<bool> visited_v(n, false);
        std::vector<bool> visited_u(n, false);

        int marked_i = i, marked_j = -1;
        std::vector<int> links(n, -1);

        while (true) 
        {
            visited_u[marked_i] = true;
            int j = -1;
            float delta = INF;

            for (int k = 0; k < n; ++k) 
            {
                if (!visited_v[k]) 
                {
                    float cur = cost_padded.at<float>(marked_i, k) - u[marked_i] - v[k];

                    if (cur < min_slack[k]) 
                    {
                        min_slack[k] = cur;
                        prev[k] = marked_j;
                        links[k] = marked_i;
                    }
                    
                    if (min_slack[k] < delta) 
                    {
                        delta = min_slack[k];
                        j = k;
                    }
                }
            }

            // Dual variable updates
            for (int k = 0; k < n; ++k)
            {
                if (visited_u[k]) 
                {    
                    u[k] += delta;
                }
            }

            for (int k = 0; k < n; ++k) 
            {
                if (visited_v[k]) 
                {
                    v[k] -= delta;
                } else
                { 
                    min_slack[k] -= delta;
                }
            }

            visited_v[j] = true;
            marked_j = j;
            marked_i = match_v[j];

            if (marked_i == -1) break; // augmenting path found
        }

        // update 
        while (marked_j != -1) 
        {
            int i_prev = links[marked_j];
            int next_j = assignment[i_prev];
            match_v[marked_j] = i_prev;
            assignment[i_prev] = marked_j;
            marked_j = next_j;
        }
    }

    return assignment;

}

static cv::Mat computeCostMatrix(const std::vector<Track>& tracks, const std::vector<Box>& detects)
{
    int w = detects.size();
    int h = tracks.size();

    cv::Mat cost_mat(cv::Size(w, h), CV_32F);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            cost_mat.at<float>(i, j) = euclidianDistance(detects[j].xywh_.x, detects[j].xywh_.y, tracks[i].get_xhat(), tracks[i].get_yhat());
        }
    }

    return cost_mat;
}

Tracker::Tracker() {}
Tracker::~Tracker() {}

void Tracker::estimateAllTracks() 
{
    for (auto& track : tracks_)
    {
        track.kf_.predict();
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

    constexpr float threshold = 30.0f;
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
                tracks_[i].correction(detects[label]);
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
