#include "trt_engine.hpp"
#include "utils.hpp"
#include "yolov8.hpp"
#include "sort_tracker.hpp"

#include <NvInfer.h>

#include <iostream>
#include <fstream>

using namespace std;

const string assets_root = "/home/alper/projects/vision-engine/scripts/";

const string video_name = "traffic3.mp4";
const string image_name = "highway.jpg";
const string model_name = "yolov8n.plan";

const string video_path = assets_root + video_name;
const string image_path = assets_root + image_name;
const string model_path = assets_root + model_name;

int main()    
{    
    std::cout << "TensorRT version: "
        << NV_TENSORRT_MAJOR << "."
        << NV_TENSORRT_MINOR << "."
        << NV_TENSORRT_PATCH << std::endl;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }

    Yolov8 nn(model_path);
    Tracker tracker;

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
                drawTrack(frame, track);
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