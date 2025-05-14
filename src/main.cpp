#include "trt_engine.hpp"
#include "utils.hpp"
#include "yolov8.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace nvinfer1;

int main()    
{   
    const string assets_root = "C:/VSCode_Repo/Real Upscale/assets/";
    const string image_name = "highway.jpg";
    const string image_path = assets_root + image_name;
    const string path = "C:/VSCode_Repo/Real Upscale/assets/yolov8n.plan";
        
    cv::Mat input;
    input = cv::imread(image_path);

    cv::Rect roi(0, 0, 640, 640);
    cv::Mat input_crop = input(roi).clone();
    
    Yolov8 nn;

    auto start = timer::now();
        
    std::vector<Box> boxes = nn.infer(input_crop);
    
    auto end = timer::now();
    printTime("infer time", start, end);

    draw_boxes(input_crop, boxes);

    cv::imwrite("output2.png", input_crop);
}
