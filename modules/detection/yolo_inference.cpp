#include "yolo_inference.hpp"

Yolov8::Yolov8(const YoloConfig &cfg, std::unique_ptr<IEngine> engine)
    : cfg_(cfg), engine_(std::move(engine))
{
    engine_->init(cfg_.engine_path);
    input_size_ = cfg_.input_width * cfg_.input_height * 3; // assuming 3 channels (BGR or RGB)

    // YOLOv8 uses three detection feature maps (stride 8, 16, 32)
    // Example for 640x640 input => (80x80 + 40x40 + 20x20) = 8400 total cells
    output_cell_count_ =
        (cfg_.input_width / 8) * (cfg_.input_height / 8) +   // P3
        (cfg_.input_width / 16) * (cfg_.input_height / 16) + // P4
        (cfg_.input_width / 32) * (cfg_.input_height / 32);  // P5

    // Each prediction per cell: [x, y, w, h] + class scores
    preds_per_cell_ = cfg_.class_num + 4;

    // Total output tensor size = total cells × predictions per cell
    output_size_ = output_cell_count_ * preds_per_cell_;
    preproc_buffer_ = new unsigned char[input_size_];
}

std::vector<Box> Yolov8::infer(const cv::Mat &input)
{
    // Preprocess
    letterbox_t letter = {0, 0, 1.0f};
    cv::Mat input_blob = preprocess(input, letter);

    // Set input buffers
    CV_Assert(input_blob.type() == CV_32FC1 && input_blob.isContinuous());

    // Inference
    engine_->set_input(input_blob.ptr<float>());
    engine_->infer();
    float *output_buf = (float *)engine_->get_output();
    
    // Postprocess
    std::vector<Box> detections = postprocess(cv::Mat(preds_per_cell_, output_cell_count_, CV_32F, output_buf),
                                              letter,
                                              input.rows,
                                              input.cols);

    nms(detections, cfg_.nms_threshold, false);
    return detections;
}

cv::Mat Yolov8::preprocess(const cv::Mat &input, letterbox_t &letter)
{
    // resize & padding if necessary
    ImageWrapper img_wrapper;
    img_wrapper.data = input.data;
    img_wrapper.width = input.cols;
    img_wrapper.height = input.rows;
    img_wrapper.channels = input.channels();
    letterbox(img_wrapper, preproc_buffer_, letter, cfg_.input_width, cfg_.input_height);

    // nhwc to nchw
    cv::Mat tensor = cv::dnn::blobFromImage(cv::Mat(cv::Size(cfg_.input_width, cfg_.input_height), CV_8UC3, preproc_buffer_),
                                            1.0 / 255.0,
                                            cv::Size(),
                                            cv::Scalar(),
                                            true,
                                            false,
                                            CV_32F);
    return tensor;
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

    float in_w = cfg_.input_width;
    float in_h = cfg_.input_height;
    float conf_threshold = cfg_.conf_threshold;

    float *__restrict output_ptr = yolo_outputT.ptr<float>();
    for (int i = 0; i < output_cell_count_; ++i)
    {
        int offset = preds_per_cell_ * i;
        valid = false;
        for (int j = 4; j < preds_per_cell_; ++j)
        {
            float score = output_ptr[offset + j];
            if (score > conf_threshold)
            {
                max_score = score;
                max_id = j - 4;
                valid = true;
            }
        }

        if (valid)
        {
            float x = output_ptr[offset] * in_w - letter.x_pad;
            float y = output_ptr[offset + 1] * in_h - letter.y_pad;
            float w = output_ptr[offset + 2] * in_w;
            float h = output_ptr[offset + 3] * in_h;

            int x1 = std::clamp(static_cast<int>((x - w / 2.0f) / letter.scale), 0, image_w);
            int y1 = std::clamp(static_cast<int>((y - h / 2.0f) / letter.scale), 0, image_h);
            int x2 = std::clamp(static_cast<int>((x + w / 2.0f) / letter.scale), 0, image_w);
            int y2 = std::clamp(static_cast<int>((y + h / 2.0f) / letter.scale), 0, image_h);

            output_boxes.emplace_back(x1, y1, x2, y2, max_id, max_score);
            ++valid_box_num;
        }
    }

    return output_boxes;
}
