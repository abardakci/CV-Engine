#include "trt_engine.hpp"
#include "yolo_detector.hpp"
#include "timer.hpp"
#include "drawer.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iostream>
#include <fstream>

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
        
        tracker.SORT(boxes);

        for (auto& track : tracker.tracks_)
        {
            if (track.age_ == 0)
                drawBox(frame, track);
        }

        std::cout << "Active tracks: " << tracker.tracks_.size() << std::endl;

        auto end = timer::now();
        printTime("total tracker time", start, end);

        cv::imshow("Tracker", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}