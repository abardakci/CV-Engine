#include "yolov8.hpp"

Yolov8::Yolov8(const std::string& path) : m_trt_engine(path)
{

}

Yolov8::~Yolov8()
{

}

std::vector<Box> Yolov8::infer(const cv::Mat& input)
{    
    letterbox_t letter = {0, 0, 1.0f};
    cv::Mat input_blob = pre_process(input, letter);

    CV_Assert(input_blob.type() == CV_32FC1 && input_blob.isContinuous());

    float* input_data = reinterpret_cast<float*>(input_blob.data);

    std::vector<float> output(m_trt_engine.m_output_size);

    m_trt_engine.infer(input_data, output.data());
    
    std::vector<Box> out = post_process(cv::Mat(num_of_class + 4, 8400, CV_32F, output.data()), letter, input.rows, input.cols);

    return out;
}

cv::Mat Yolov8::pre_process(const cv::Mat& input, letterbox_t& letter)
{
    // resize & padding if necessary
    cv::Mat input_letter = letterbox(input, letter, 640, 640);  

    // nhwc  to nchw
    cv::Mat input_blob = cv::dnn::blobFromImage(input_letter);
    
    // uchar to fp32&normalization
    cv::Mat output;
    input_blob.convertTo(output, CV_32F, 1.0f/255.0f);

    return output;
}

std::vector<Box> Yolov8::post_process(const cv::Mat& yolo_output, letterbox_t& letter, int image_h, int image_w)
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
            float x = yolo_outputT.at<float>(i, 0) * 640.0f - letter.x_pad;
            float y = yolo_outputT.at<float>(i, 1) * 640.0f - letter.y_pad;
            float w = yolo_outputT.at<float>(i, 2) * 640.0f;
            float h = yolo_outputT.at<float>(i, 3) * 640.0f;

            int x1 = clamp(static_cast<int>((x - w / 2.0f) / letter.scale), 0, image_w);
            int y1 = clamp(static_cast<int>((y - h / 2.0f) / letter.scale), 0, image_h);
            int x2 = clamp(static_cast<int>((x + w / 2.0f) / letter.scale), 0, image_w);
            int y2 = clamp(static_cast<int>((y + h / 2.0f) / letter.scale), 0, image_h);
            
            output_boxes.emplace_back(x1, y1, x2, y2, max_id, max_score);
            ++valid_box_num;
        }
    }
    
    return output_boxes; 
}
