#include "sort_cfg.hpp"
#include <yaml-cpp/yaml.h>

SORTConfig loadSORTConfig(const std::string &config_path)
{
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config["sort_tracker"])
        throw std::runtime_error("[" + config_path + "] Missing 'sort_tracker' section in YAML.");

    const YAML::Node &node = config["sort_tracker"];
    SORTConfig cfg;
    auto safeGet = [&](const std::string &key, auto default_value)
    {
        if (node[key])
            return node[key].as<std::decay_t<decltype(default_value)>>();
        return default_value;
    };

    cfg.demo_video_path = safeGet("demo_video", std::string{});
    cfg.assignment_threshold = safeGet("assignment_threshold", 25.0f);
    cfg.max_age = safeGet("max_age", 10);
    return cfg;
}
