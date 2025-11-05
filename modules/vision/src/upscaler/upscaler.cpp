#include "upscaler.hpp"
#include "engine_interface.hpp"

Upscaler::Upscaler(std::unique_ptr<IEngine> engine, const std::string& engine_path) : engine_(std::move(engine)) 
{
    engine_->init(engine_path);
}

cv::Mat Upscaler::run(cv::Mat &input)
{
    auto ten = preproc(input);
    engine_->set_input(ten.ptr<float>());
    engine_->infer();
    float* buf = engine_->get_output(); 
}

cv::Mat Upscaler::preproc(cv::Mat &mat)
{
    CV_Assert(mat.type() == CV_8UC3 && mat.isContinuous());

    // nhwc to nchw
    cv::Mat tensor = cv::dnn::blobFromImage(mat,
                                            1.0 / 255.0,
                                            cv::Size(),
                                            cv::Scalar(),
                                            true,
                                            false,
                                            CV_32F);

    return tensor;
}

cv::Mat Upscaler::postproc(cv::Mat &tensor)
{
    // tensor: 1x3xHxW, CV_32F
    CV_Assert(tensor.type() == CV_32F);

    int batch = tensor.size[0];
    int channels = tensor.size[1];
    int height = tensor.size[2];
    int width = tensor.size[3];
    CV_Assert(batch == 1 && channels == 3);

    // NCHW -> CHW
    std::vector<cv::Mat> ch(3);
    for (int c = 0; c < 3; c++)
    {
        ch[c] = cv::Mat(height, width, CV_32F, (float*)tensor.ptr<float>(0, c));
    }

    // CHW -> HWC
    cv::Mat rgb;
    cv::merge(ch, rgb);   // HxWx3, float32 [0,1]

    // RGB -> BGR
    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);

    // 0-1 -> 0-255, uint8
    cv::Mat out;
    rgb.convertTo(out, CV_8UC3, 255.0);

    return out;
}
