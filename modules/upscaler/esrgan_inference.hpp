// #pragma once

// #include <iostream>
// #include <algorithm>
// #include <opencv2/opencv.hpp>

// #include "engine_interface.hpp"
// #include "types.hpp"
// #include "postproc.hpp"
// #include "preproc.hpp"
// #include "timer.hpp"

// class ESRGAN
// {
// public:
//     ESRGAN(std::unique_ptr<IEngine> engine);
//     ~ESRGAN() = default;
//     cv::Mat infer(const cv::Mat &input);

// private:
//     std::unique_ptr<IEngine> engine_;
//     size_t input_size_;
//     size_t output_size_;
//     std::array<float*, N_MAX_INPUT> input_buf_;
//     std::array<float*, N_MAX_OUTPUT> output_buf_;
// };
