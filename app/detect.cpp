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
    // exe path
    namespace fs = std::filesystem;
    fs::path exe_path;
    try {
        exe_path = fs::canonical(argv[0]);  // tam path
    } catch (...) {
        exe_path = fs::current_path() / argv[0];
    }

    // binary dizini (out/install/debug/bin)
    fs::path exe_dir = exe_path.parent_path();

    // config dosyası konumu (../share/config/config.yaml)
    fs::path config_path = exe_dir / "../share/config/config.yaml";
    config_path = fs::weakly_canonical(config_path); // normalize
    
    YAML::Node config = YAML::LoadFile(config_path.string());
    std::string video_path = config["assets"]["video_path"].as<string>();
    std::string engine_path = config["assets"]["engine_path"].as<string>();

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
        print_time("Inference time", start, end);
        
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
