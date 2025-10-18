#include "utils.hpp"

cv::Mat letterbox(cv::Mat input, letterbox_t& letter, int w, int h)
{
    if (input.rows == h && input.cols == w)
    {
        return input;
    }

    cv::Mat dst = cv::Mat::zeros(cv::Size(w, h), input.type());

    int in_h = input.rows;
    int in_w = input.cols;
    float scale = std::min(static_cast<float>(w) / in_w, static_cast<float>(h) / in_h);
    letter.scale = scale;
    
    if (in_h > in_w)
    {
        int w_out = in_w * scale;

        int pad_per_side = (640 - w_out) / 2;
        letter.x_pad = pad_per_side;

        cv::Rect roi(pad_per_side, 0, w_out, 640);
        cv::Mat roi_dst = dst(roi); 
        cv::resize(input, roi_dst, roi_dst.size(), scale, scale);
    }

    else if (in_w > in_h)
    {
        int h_out = in_h * scale;

        int pad_per_side = (640 - h_out) / 2;  
        letter.y_pad = pad_per_side; 
        
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

void drawBox(cv::Mat& target, const Box& box) 
{
    // Renk ve font ayarları
    const cv::Scalar color(0, 120, 0); // Yeşil (BGR formatında)
    const int thickness = 1;
    const float font_scale = 0.3;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;

    // Dikdörtgen çiz
    cv::Point pt1(box.xyxy_.x1, box.xyxy_.y1);
    cv::Point pt2(box.xyxy_.x2, box.xyxy_.y2);
    cv::rectangle(target, pt1, pt2, color, thickness);

    // Etiket metni (class_id + confidence)
    std::string label = "Class " + std::to_string(box.class_id_) + 
                        " (" + std::to_string(box.conf_score_).substr(0, 4) + ")";

    // Etiket için arka plan dikdörtgeni
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, 1, &baseline);
    cv::Point text_org(box.xyxy_.x1, box.xyxy_.y1 - 5); // Üstte küçük boşluk bırak

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

void drawBoxes(cv::Mat& target, const std::vector<Box>& boxes)
{
    for (auto box : boxes)
    {
        drawBox(target, box);
    }
}

void drawTrack(cv::Mat& target, Track& track)
{
    const cv::Scalar color(150, 0, 0); // Yeşil (BGR formatında)
    const int thickness = 1;
    const float font_scale = 0.3;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;

    // Dikdörtgen çiz
    cv::Point pt1(track.bbox_.xyxy_.x1, track.bbox_.xyxy_.y1);
    cv::Point pt2(track.bbox_.xyxy_.x2, track.bbox_.xyxy_.y2);
    cv::rectangle(target, pt1, pt2, color, thickness);

    std::string label = "Track id " + std::to_string(track.track_id_) + 
                        " (" + std::to_string(track.bbox_.conf_score_).substr(0, 4) + ")";

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, 1, &baseline);
    cv::Point text_org(track.bbox_.xyxy_.x1, track.bbox_.xyxy_.y1 - 5); // Üstte küçük boşluk bırak

    cv::rectangle(
        target, 
        cv::Rect(text_org.x, text_org.y - text_size.height, 
                text_size.width, text_size.height + baseline), 
        color, 
        cv::FILLED
    );

    cv::putText(
        target, 
        label, 
        text_org, 
        font_face, 
        font_scale, 
        cv::Scalar(255, 255, 255),
        1, 
        cv::LINE_AA
    );
}

float iou(Box &b1, Box& b2)
{
    if (b1.xyxy_.x1 > b2.xyxy_.x2 || b1.xyxy_.x2 < b2.xyxy_.x1 ||
        b1.xyxy_.y1 > b2.xyxy_.y2 || b1.xyxy_.y2 < b2.xyxy_.y1)
    {
        return 0.0f;
    }

    int x_left   = std::max(b1.xyxy_.x1, b2.xyxy_.x1);
    int y_left   = std::max(b1.xyxy_.y1, b2.xyxy_.y1);
    int x_right  = std::min(b1.xyxy_.x2, b2.xyxy_.x2);
    int y_right  = std::min(b1.xyxy_.y2, b2.xyxy_.y2);

    int inter_width = x_right - x_left;
    int inter_height = y_right - y_left;

    int inter_area = inter_width * inter_height;

    int b1_area = (b1.xyxy_.x2 - b1.xyxy_.x1) * (b1.xyxy_.y2 - b1.xyxy_.y1);
    int b2_area = (b2.xyxy_.x2 - b2.xyxy_.x1) * (b2.xyxy_.y2 - b2.xyxy_.y1);

    int union_area = b1_area + b2_area - inter_area;

    if (union_area == 0) 
    {
        return 0.0f;
    }

    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

static bool compareBoxes(const Box& b1, const Box& b2)
{
    return b1.conf_score_ > b2.conf_score_;
}

void nms(std::vector<Box>& bboxes, float iou_threshold, bool is_sorted)
{
    if (!is_sorted)
        std::sort(bboxes.begin(), bboxes.end(), compareBoxes);

    
    std::vector<Box> result_boxes;
    std::vector<bool> suppressed(bboxes.size(), false);
    
    for (int i = 0; i < bboxes.size(); ++i)
    {
        if (suppressed[i]) 
            continue;

        result_boxes.push_back(bboxes[i]);

        for (int j = i + 1; j < bboxes.size(); ++j)
        {
            if (suppressed[j]) 
            {
                continue;
            }

            float current_iou = iou(bboxes[i], bboxes[j]);

            if (current_iou > iou_threshold) 
            {
                suppressed[j] = true;
            }
        }
    }

    bboxes = result_boxes;
}
