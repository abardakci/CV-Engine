#include "yolo_detector.hpp"

Yolov8::Yolov8(const std::string &path, std::unique_ptr<IEngine> engine) : engine_(std::move(engine))
{
    engine_->initialize(path);
}

Yolov8::~Yolov8() {}

std::vector<Box> Yolov8::infer(const cv::Mat &input)
{
    letterbox_t letter = {0, 0, 1.0f};
    cv::Mat input_blob = preprocess(input, letter);

    CV_Assert(input_blob.type() == CV_32FC1 && input_blob.isContinuous());
    
    float *input_data = reinterpret_cast<float *>(input_blob.data);
    std::vector<float*> inputs;
    std::vector<float*> outputs;
    inputs.push_back(input_data);

    std::vector<float> output(output_size_);
    outputs.push_back(output.data());
    engine_->infer(inputs, outputs);
    
    std::vector<Box> detections = postprocess(cv::Mat(kClassNum + 4, 8400, CV_32F, output.data()), letter, input.rows, input.cols);
    nms(detections, kNmsThreshold, false);
    return detections;
}

cv::Mat Yolov8::preprocess(const cv::Mat &input, letterbox_t &letter)
{
    // resize & padding if necessary
    cv::Mat input_letter = letterbox(input, letter, 640, 640);

    // nhwc to nchw
    cv::Mat input_blob = cv::dnn::blobFromImage(input_letter);

    // uchar to fp32&normalization
    cv::Mat output;
    input_blob.convertTo(output, CV_32F, 1.0f / 255.0f);

    return output;
}

std::vector<Box> Yolov8::postprocess(const cv::Mat &yolo_output, letterbox_t &letter, int image_h, int image_w)
{
    std::vector<Box> output_boxes;
    output_boxes.reserve(100);

    cv::Mat yolo_outputT;
    cv::transpose(yolo_output, yolo_outputT);

    int valid_box_num = 0;
    bool valid = false;
    int max_id = 0;
    float max_score = 0.0f;

    float *__restrict output_ptr = yolo_outputT.ptr<float>();
    for (int i = 0; i < 8400; ++i)
    {
        int offset = 84 * i;
        valid = false;
        for (int j = 4; j < 84; ++j)
        {
            float score = output_ptr[offset + j];
            if (score > kConfThreshold)
            {
                max_score = score;
                max_id = j - 4;
                valid = true;
            }
        }

        if (valid)
        {
            float x = output_ptr[offset] * 640.0f - letter.x_pad;
            float y = output_ptr[offset + 1] * 640.0f - letter.y_pad;
            float w = output_ptr[offset + 2] * 640.0f;
            float h = output_ptr[offset + 3] * 640.0f;

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
