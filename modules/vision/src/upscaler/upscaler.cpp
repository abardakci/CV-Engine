#include "upscaler.hpp"
#include "engine_interface.hpp"

Upscaler::Upscaler(std::unique_ptr<IEngine> engine, const std::string &engine_path) : engine_(std::move(engine))
{
    engine_->init(engine_path);
}

cv::Mat Upscaler::run(cv::Mat &input)
{
    CV_Assert(input.type() == CV_8UC3 && input.isContinuous());

    // nhwc to nchw
    cv::Mat tensor = cv::dnn::blobFromImage(input,
                                            1.0 / 255.0,
                                            cv::Size(),
                                            cv::Scalar(),
                                            true,
                                            false,
                                            CV_32F);
    tensor::Dims shape;
    shape.d[0] = tensor.size[0];
    shape.d[1] = tensor.size[1];
    shape.d[2] = tensor.size[2];
    shape.d[3] = tensor.size[3];

    engine_->set_input_shape(0, shape);
    engine_->set_input(tensor.ptr<float>());
    engine_->infer();
    float *buf = engine_->get_output();

    int C = shape.d[1];
    int H = shape.d[2] * 2;
    int W = shape.d[3] * 2;

    std::vector<cv::Mat> ch(C);
    for (int c = 0; c < C; ++c)
    {
        ch[c] = cv::Mat(H, W, CV_32F, buf + c * H * W);
    }

    cv::Mat rgb;
    cv::merge(ch, rgb); // HxWxC

    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);

    cv::Mat out;
    rgb.convertTo(out, CV_8UC3, 255.0);

    return out;
}
