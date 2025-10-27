#pragma once

#include <iostream>
#include <unordered_map>

struct YoloConfig
{
    // EngineType engine_type;
    std::string engine_path;
    std::string engine_path_fp16;
    std::string demo_video_path;
    bool fp16_available;
    float nms_threshold;
    float conf_threshold;
    int class_num;
    int input_width;
    int input_height;
};

YoloConfig loadYoloConfig(const std::string &config_path);
