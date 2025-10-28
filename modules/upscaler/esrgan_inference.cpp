#include "esrgan_inference.hpp"
#include <iostream>

ESRGAN::ESRGAN(std::unique_ptr<IEngine> engine)
    : engine_(std::move(engine)),
      input_size_(0),
      output_size_(0)
{
    input_size_ = 960 * 540 * 3;
    output_size_ = input_size_ * 2 * 2;
}

cv::Mat ESRGAN::infer(const cv::Mat &input)
{
    engine_->infer(input_buf_, output_buf_);
}
