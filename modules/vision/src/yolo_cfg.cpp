#include "yolo_cfg.hpp"
#include <yaml-cpp/yaml.h>

YoloConfig loadYoloConfig(const std::string &config_path)
{
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config["yolo"])
        throw std::runtime_error("[" + config_path + "] Missing 'yolo' section in YAML.");

    const YAML::Node &yolo = config["yolo"];
    YoloConfig cfg;

    // static const std::unordered_map<std::string, EngineType> kEngineMap = {
    //     {"TensorRT", EngineType::TensorRT},
    //     {"OnnxRuntime", EngineType::OnnxRuntime}};

    // std::string engine_name = yolo["engine_type"].as<std::string>();
    // if (!kEngineMap.contains(engine_name))
    //     throw std::invalid_argument("Unknown engine_type: " + engine_name);
    // cfg.engine_type = kEngineMap.at(engine_name);

    auto safeGet = [&](const std::string &key, auto default_value)
    {
        if (yolo[key])
            return yolo[key].as<std::decay_t<decltype(default_value)>>();
        return default_value;
    };

    cfg.fp16_available = safeGet("fp16_available", false);
    cfg.engine_path_fp16 = safeGet("engine_path_fp16", std::string{});
    cfg.engine_path = safeGet("engine_path", std::string{});
    cfg.demo_video_path = safeGet("demo_video", std::string{});
    cfg.nms_threshold = safeGet("nms_threshold", 0.4f);
    cfg.conf_threshold = safeGet("conf_threshold", 0.25f);
    cfg.class_num = safeGet("class_num", 80);
    cfg.input_width = safeGet("input_width", 640);
    cfg.input_height = safeGet("input_height", 640);

    return cfg;
}
