#include "yolo_detector.hpp"
#include "sort_tracker.hpp"
#include "kf_wrapper.hpp"
#include "timer.hpp"
#include "drawer.hpp"
#include "engine_factory.hpp"
#include "yolo_cfg.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char** argv)    
{       
    std::string video_path;
    bool use_fp16 = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--video" && i + 1 < argc)
            video_path = argv[++i];
        else if (std::string(argv[i]) == "--fp16")
            use_fp16 = true;
        else if (std::string(argv[i]) == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [--video <video_path>] [--fp16]" << std::endl;
            return 0;
        }
    }

    namespace fs = std::filesystem;
    fs::path config_path = fs::current_path() / "share/config/config.yaml";
    YoloConfig cfg = loadYoloConfig(config_path.string());

    if (video_path.empty())
        video_path = cfg.demo_video_path;

    if (use_fp16 && cfg.fp16_available)
    {
        std::cout << "Using FP16 precision for inference." << std::endl;
        cfg.engine_path = cfg.engine_path_fp16;
    }
    else
    {
        std::cout << "Using FP32 precision for inference." << std::endl;
        cfg.engine_path = cfg.engine_path;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }
    
    std::unique_ptr<IEngine> engine = EngineFactory().create(EngineType::TensorRT);
    Yolov8 nn(cfg, std::move(engine));
    Tracker sort(KalmanType::KFWrapper);

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty()) break;

        auto start = timer::now();

        vector<Box> boxes = nn.infer(frame);

        sort.SORT(boxes);

        std::cout << "Active tracks: " << sort.tracks_.size() << std::endl;

        auto end = timer::now();
        print_time("total tracker time", start, end);

        for (const auto& track : sort.tracks_)
        {
            if (track.age_ == 0)
                drawBox(frame, track.bbox_);
        }
 
        cv::imshow("Tracker", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}