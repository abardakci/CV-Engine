#include "vision_toolkit/preproc.hpp"
#include <opencv2/opencv.hpp>

void letterbox(ImageWrapper input, unsigned char* output_buffer, letterbox_t &letter, int target_width, int target_height)
{
    if (input.width == target_width && input.height == target_height)
    {
        return;
    }

    cv::Mat dst(target_width, target_height, CV_8UC3, output_buffer);
    cv::Mat input_mat(input.height, input.width, CV_8UC3, input.data);

    int in_h = input.height;
    int in_w = input.width;
    float scale = std::min(static_cast<float>(target_width) / in_w, static_cast<float>(target_height) / in_h);
    letter.scale = scale;

    if (in_h > in_w)
    {
        int w_out = in_w * scale;

        int pad_per_side = (640 - w_out) / 2;
        letter.x_pad = pad_per_side;

        cv::Rect roi(pad_per_side, 0, w_out, 640);
        cv::Mat roi_dst = dst(roi);
        cv::resize(input_mat, roi_dst, roi_dst.size(), scale, scale);
    }

    else if (in_w > in_h)
    {
        int h_out = in_h * scale;

        int pad_per_side = (640 - h_out) / 2;
        letter.y_pad = pad_per_side;

        cv::Rect roi(0, pad_per_side, 640, h_out);
        cv::Mat roi_dst = dst(roi);
        cv::resize(input_mat, roi_dst, roi_dst.size(), scale, scale);
    }
}
