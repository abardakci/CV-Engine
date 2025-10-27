#pragma once

#include <vector>
#include "types.hpp"
#include "kalman_interface.hpp"
#include "kalman_factory.hpp"
#include "sort_cfg.hpp"

static constexpr int num_states_kf = 4;
static constexpr int num_measures_kf = 2;

class Tracker
{
public:
    Tracker(SORTConfig sort_cfg, KalmanType kalman_type);
    ~Tracker();

    void estimateAllTracks();
    void addNewTrack(Box& new_box);
    void removeOldTracks(const int age_limit = 10);
    void SORT(std::vector<Box>& detects);

    std::vector<Track> tracks_;

private:
    KalmanType kf_type_;
    long highest_id_ = 0;
    SORTConfig cfg_;

};
