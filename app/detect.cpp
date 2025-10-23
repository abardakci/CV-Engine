#include "engine.hpp"
#include "yolo_detector.hpp"
#include "timer.hpp"
#include "drawer.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <format>

using namespace std;

int main(int argc, char** argv)    
{       
    // std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
    // std::filesystem::path config_path = exe_path.parent_path() / "config/config.yaml";

    // YAML::Node config = YAML::LoadFile(config_path.string());
    // std::string video_path = config["assets"]["demo_video"].as<string>();
    // std::string engine_path = config["yolo"]["engine_path"].as<string>();
    std::string video_path  = "/home/alper/projects/vision_engine/install/assets/video-image/traffic3.mp4";
    std::string engine_path = "/home/alper/projects/vision_engine/install/assets/models/yolov8n.plan";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }

    auto trt = std::make_unique<TrtEngine>();    
    Yolov8 nn(engine_path, std::move(trt));

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty()) break;

        auto start = timer::now();

        vector<Box> boxes = nn.infer(frame);
        
        auto end = timer::now();
        printTime("Inference time", start, end);
        
        for (auto &box : boxes)
        {
            string s = std::format("Class: {}, Conf: {}", box.class_id_, box.conf_score_);
            drawBox(frame, box, s);
        }
        
        cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
