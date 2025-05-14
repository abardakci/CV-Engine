#include "yolov8.hpp"

Yolov8::Yolov8() : m_trt_engine("C:/VSCode_Repo/Real Upscale/assets/yolov8n.plan")
{

}

Yolov8::~Yolov8()
{

}

cv::Mat Yolov8::pre_process(const cv::Mat& input)
{ 
    // nhwc  to nchw
    cv::Mat input_;
    input_ = cv::dnn::blobFromImage(input);
    
    // uchar to fp32&normalization
    cv::Mat output;
    input_.convertTo(output, CV_32F, 1.0f/255.0f);

    return output;
}

std::vector<Box> Yolov8::post_process(const cv::Mat& yolo_output)
{
    std::vector<Box> output_boxes;

    cv::Mat yolo_outputT;
    cv::transpose(yolo_output, yolo_outputT);

    int valid_box_num = 0;
    bool valid = false;
    int max_id = 0;
    float max_score = 0.0f;

    for (int i = 0; i < 8400; ++i)
    {
        valid = false;
        for (int j = 4; j < 84; ++j)
        {
            float score = yolo_outputT.at<float>(i, j);   
            if (score > conf_threshold)
            {
                max_score = score;
                max_id = j - 4;
                valid = true;
            }
        }

        if (valid)
        {
            float x = yolo_outputT.at<float>(i, 0) * 640;
            float y = yolo_outputT.at<float>(i, 1) * 640;
            float w = yolo_outputT.at<float>(i, 2) * 640;
            float h = yolo_outputT.at<float>(i, 3) * 640;

            int x1 = static_cast<int>(x - w / 2.0f);
            int y1 = static_cast<int>(y - h / 2.0f);
            int x2 = static_cast<int>(x + w / 2.0f);
            int y2 = static_cast<int>(y + h / 2.0f);
            
            std::cout << x << " x\n";

            output_boxes.emplace_back(x1, y1, x2, y2, max_id, max_score);
            ++valid_box_num;
        }
    }

    std::cout << "Valid box number: " << valid_box_num << std::endl;
    
    return output_boxes; 
}

std::vector<Box> Yolov8::infer(const cv::Mat& input)
{   
    cv::Mat input_ = pre_process(input);

    CV_Assert(input_.type() == CV_32FC1 && input_.isContinuous());

    float* input_data = reinterpret_cast<float*>(input_.data);

    std::vector<float> output(m_trt_engine.m_output_size);
    
    m_trt_engine.infer(input_data, output.data());

    return post_process(cv::Mat(num_of_class + 4, 8400, CV_32F, output.data()));
}
