#include "utils.hpp"

cv::Mat letterbox(cv::Mat input, int w, int h)
{
    if (input.rows == h && input.cols == w)
    {
        return input;
    }

    cv::Mat dst = cv::Mat::zeros(cv::Size(w, h), input.type());

    int in_h = input.rows;
    int in_w = input.cols;
    float scale = std::min(static_cast<float>(w) / in_w, static_cast<float>(h) / in_h);

    if (in_h > in_w)
    {
        int w_out = in_w * scale;

        int pad_per_side = (640 - w_out) / 2;  
        cv::Rect roi(pad_per_side, 0, w_out, 640);
        cv::Mat roi_dst = dst(roi); 
        cv::resize(input, roi_dst, roi_dst.size(), scale, scale);
    }

    else if (in_w > in_h)
    {
        int h_out = in_h * scale;

        int pad_per_side = (640 - h_out) / 2;  
        cv::Rect roi(0, pad_per_side, 640, h_out);
        cv::Mat roi_dst = dst(roi); 
        cv::resize(input, roi_dst, roi_dst.size(), scale, scale);
    }

    return dst;
}

void printTime(const std::string &msg, timer::time_point start, timer::time_point end)
{
    std::cout << msg << " --- " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
}

int loadBinaryFromFile(const std::string& fileName, std::vector<char>& output)
{
    std::ifstream reader(fileName, std::ios::binary | std::ios::ate);
    if (!reader) 
    {
        std::cerr << "File not open error: " << fileName << std::endl;
        return -1;
    }

    size_t size = reader.tellg();
    if (size <= 0) 
    {
        std::cerr << "File size invalid error: " << fileName << std::endl;
        size = 0;
        return -1;
    }

    output.resize(size);

    reader.seekg(0, std::ios::beg);
    if (!reader.read(output.data(), size)) 
    {
        std::cerr << "File read error: " << fileName << std::endl;
        return -1;
    }

    return 0;
}

void draw_box(cv::Mat& target, const Box& box) 
{
    // Renk ve font ayarları
    const cv::Scalar color(0, 120, 0); // Yeşil (BGR formatında)
    const int thickness = 1;
    const float font_scale = 0.3;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;

    // Dikdörtgen çiz
    cv::Point pt1(box.x1, box.y1);
    cv::Point pt2(box.x2, box.y2);
    cv::rectangle(target, pt1, pt2, color, thickness);

    // Etiket metni (class_id + confidence)
    std::string label = "Class " + std::to_string(box.class_id) + 
                        " (" + std::to_string(box.conf_score).substr(0, 4) + ")";

    // Etiket için arka plan dikdörtgeni
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, 1, &baseline);
    cv::Point text_org(box.x1, box.y1 - 5); // Üstte küçük boşluk bırak

    // Arka planı çiz
    cv::rectangle(
        target, 
        cv::Rect(text_org.x, text_org.y - text_size.height, 
                text_size.width, text_size.height + baseline), 
        color, 
        cv::FILLED
    );

    // Metni yaz
    cv::putText(
        target, 
        label, 
        text_org, 
        font_face, 
        font_scale, 
        cv::Scalar(0, 0, 0), // Siyah metin
        1, 
        cv::LINE_AA
    );
}

void draw_boxes(cv::Mat& target, const std::vector<Box>& boxes)
{
    for (auto box : boxes)
    {
        draw_box(target, box);
    }
}