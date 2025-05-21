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
    void SORT(std::vector<Box>& detects);

private:
    std::vector<Track> tracks_;
    
};
