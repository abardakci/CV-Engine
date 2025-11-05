#include "yolo_runner.hpp"

#include "upscaler/upscaler.hpp"
#include "engine_factory.hpp"
#include "timer.hpp"
#include "draw.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <format>

using namespace std;
using namespace forge;

int run(int argc, char **argv)
{
    std::string video_path;
    std::string engine_path;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--video" && i + 1 < argc)
            video_path = argv[++i];
        else if (std::string(argv[i]) == "--engine" && i + 1 < argc)
            video_path = argv[++i];
        else if (std::string(argv[i]) == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [--video <video_path>] [--engine <engine_path>] [--fp16]" << std::endl;
            return 0;
        }
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }
    
    auto engine = EngineFactory().create(EngineType::TensorRT);
    Upscaler upscaler(std::move(engine), engine_path);

    // Main loop
    cv::Mat frame;
    cv::Mat output;
    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        auto start = now();

        output = upscaler.run(frame);

        auto end = now();
        print_time("Inference time", start, end);
        cv::imshow("Upscaled Frame", output);
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
