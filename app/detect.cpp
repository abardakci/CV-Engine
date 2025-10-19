#include "trt_engine.hpp"
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
    std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
    std::filesystem::path config_path = exe_path.parent_path() / "config/config.yaml";

    YAML::Node config = YAML::LoadFile(config_path.string());
    std::string video_path = config["assets"]["demo_video"].as<string>();
    std::string model_path = config["assets"]["model_path"].as<string>();

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }

    Yolov8 nn(model_path);

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
