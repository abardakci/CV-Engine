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