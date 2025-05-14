#include "utils.hpp"

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

void getTiles(const cv::Mat& input, float* tiles, const int wTileNum, const int hTileNum, const int tileSize, const int overlap)
{
    // overlap = tile_size - ((w - tile_size) / tile_number)
    // stride = tile_size - overlap

    const int w = input.cols;
    const int h = input.rows;

    int wStride = (w - tileSize) / wTileNum;
    int hStride = (h - tileSize) / hTileNum;
    
    int tileByte = tileSize * tileSize * 3;
    for (int i = 0; i < hTileNum; ++i)
    {
        for (int j = 0; j < wTileNum; ++j)
        {
            float* tilePtrOffset = tiles + (i * wTileNum + j) * tileByte;
            int rowOffset = hStride * i;
            int colOffset = wStride * j;

            cv::Rect roi(colOffset, rowOffset, tileSize, tileSize);
            cv::Mat tile = input(roi).clone();
            std::memcpy(tilePtrOffset, tile.ptr<float>(), tileByte);
        }
    }

}

cv::Mat hwc2chw(const cv::Mat& hwc) 
{
    // Check if image is valid
    if (hwc.empty()) 
    {
        return cv::Mat();
    }

    // Get image dimensions
    int height = hwc.rows;
    int width = hwc.cols;
    int channels = hwc.channels();

    // Reshape to [H*W, C]
    cv::Mat flat = hwc.reshape(1, height * width);

    // Transpose to [C, H*W]
    cv::Mat chw = flat.t();

    // Reshape to [C, H, W]
    chw = chw.reshape(1, channels);

    return std::move(chw);
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