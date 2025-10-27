#pragma once

#include <iostream>
#include <unordered_map>

struct SORTConfig
{
    std::string demo_video_path;
    float assignment_threshold;
    int max_age;
};

SORTConfig loadSORTConfig(const std::string &config_path);
