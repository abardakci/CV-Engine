#include "upscaler_run.hpp"

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

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <iostream>

bool is_image(const std::string &path)
{
    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    };
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    for (auto &e : exts)
        if (lower.size() >= e.size() && lower.compare(lower.size() - e.size(), e.size(), e) == 0)
            return true;

    return false;
}

bool is_video(const std::string &path)
{
    static const std::vector<std::string> exts = {
        ".mp4", ".avi", ".mkv", ".mov", ".webm"
    };
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    for (auto &e : exts)
        if (lower.size() >= e.size() && lower.compare(lower.size() - e.size(), e.size(), e) == 0)
            return true;

    return false;
}

int run_upscaler(int argc, char **argv)
{
    std::string input_path;
    std::string engine_path;

    // Parse arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--input" && i + 1 < argc)
            input_path = argv[++i];
        else if (std::string(argv[i]) == "--engine" && i + 1 < argc)
            engine_path = argv[++i];
        else if (std::string(argv[i]) == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " --input <image_or_video> --engine <engine_path>\n";
            return 0;
        }
    }

    if (input_path.empty())
    {
        std::cerr << "[ERROR] --input <path> missing\n";
        return -1;
    }

    // Create engine
    auto engine = EngineFactory().create(EngineType::TensorRT, ShapeMode::Dynamic);
    Upscaler upscaler(std::move(engine), engine_path);

    // IMAGE MODE
    if (is_image(input_path))
    {
        cv::Mat img = cv::imread(input_path);
        if (img.empty())
        {
            std::cerr << "[ERROR] Failed to read image.\n";
            return -1;
        }

        auto start = now();
        cv::Mat out = upscaler.run(img);
        auto end = now();
        print_time("Inference", start, end);

        std::string save_path = input_path + "_x2.png";
        cv::imwrite(save_path, out);
        std::cout << "[+] Saved: " << save_path << "\n";

        return 0;
    }

    // VIDEO MODE
    if (is_video(input_path))
    {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened())
        {
            std::cerr << "[ERROR] Failed to open video.\n";
            return -1;
        }

        cv::namedWindow("Upscaled", cv::WINDOW_AUTOSIZE);

        while (true)
        {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            auto start = now();
            cv::Mat out = upscaler.run(frame);
            auto end = now();
            print_time("Frame", start, end);

            cv::imshow("Upscaled", out);

            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 27)  // ESC or q
                break;
        }

        return 0;
    }

    // INVALID EXTENSION
    std::cerr << "[ERROR] Unknown file type: " << input_path << "\n";
    return -1;
}