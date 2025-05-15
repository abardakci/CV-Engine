#include "trt_engine.hpp"
#include "utils.hpp"
#include "yolov8.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace nvinfer1;

const string assets_root = "C:/VSCode_Repo/Real Upscale/assets/";

const string image_name = "highway.jpg";
const string model_name = "yolov8n.plan";

const string image_path = assets_root + image_name;
const string model_path = assets_root + model_name;

int main()    
{           
    cv::Mat input = cv::imread(image_path);
    
    Yolov8 nn(model_path);

    auto start = timer::now();
        
    std::vector<Box> boxes = nn.infer(input);
    
    auto end = timer::now();
    printTime("infer time", start, end);

    draw_boxes(input, boxes);
    
    cv::imwrite("output.png", input);
}
