#include "yolo.hpp"

#include "yolo_detector.hpp"
#include "engine_factory.hpp"
#include "yolo_cfg.hpp"
#include "timer.hpp"
#include "drawer.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <format>

using namespace std;

int run(int argc, char **argv)
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
    Yolov8 yolo(cfg, std::move(engine));

    // Main loop
    cv::Mat frame;
    vector<Box> boxes;
    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        auto start = timer::now();

        boxes = yolo.infer(frame);

        auto end = timer::now();
        print_time("Inference time", start, end);

        for (auto &box : boxes)
        {
            string s = std::format("Class: {}, Conf: {}", box.class_id_, box.conf_score_);
            drawBox(frame, box, s);
        }

        cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
