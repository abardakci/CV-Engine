#include "trt_engine.hpp"
#include "yolo_detector.hpp"
#include "timer.hpp"
#include "drawer.hpp"

#include <NvInfer.h>

#include <iostream>
#include <fstream>

using namespace std;

const string assets_root = "/home/alper/projects/vision-engine/assets/";

const string video_name = "video-image/traffic3.mp4";
const string image_name = "video-image/highway.jpg";
const string model_name = "models/yolov8n.plan";

const string video_path = assets_root + video_name;
const string image_path = assets_root + image_name;
const string model_path = assets_root + model_name;

int main()    
{    
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

        printTime("total inference time", start, end);

        drawBoxes(frame, boxes);
        cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}