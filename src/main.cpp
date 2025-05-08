#include "trt_engine.hpp"
#include "utils.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace nvinfer1;

int main()
{    
    const string assets_root = "C:/VSCode_Repo/Real Upscale/assets/";
    const string image_name = "image2.png";

    const string imagePath = assets_root + image_name;
    const string enginePath = assets_root + "real_x2.plan"; 
    
    std::vector<char> modelBinary;
    int error = loadBinaryFromFile(enginePath, modelBinary);
    
    Logger logger;

    unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(modelBinary.data(), modelBinary.size())); 
    TrtEngine trt(*engine.get());    

    cv::Mat input;
    cv::Mat output;
    input = cv::imread(imagePath);

    int acc_time = 0;
    
    const int N = 10;
    for (int i = 0; i < N; ++i)
    {
        auto start = timer::now();
         
        trt.infer(input, output);
        
        auto end = timer::now();
        printTime("infer time", start, end);

        acc_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();
    }

    cout << "avg time: " << acc_time / N << endl;
    cv::imwrite("output.png", output);
}
