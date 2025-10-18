#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "yolov8.hpp"
#include "cv_types.hpp"

static constexpr int num_states_kf = 4;
static constexpr int num_measures_kf = 2;

class Yolov8;

class Tracker
{
public:
    Tracker();
    ~Tracker();

    void estimateAllTracks();
    void addNewTrack(Box& new_box);
    void removeOldTracks(const int age_limit = 10);
    void SORT(std::vector<Box>& detects);

    std::vector<Track> tracks_;

private:
    long highest_id_ = 0;

};
