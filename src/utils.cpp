#include "utils.hpp"

void getTiles(const cv::Mat& input, float* tiles, const int tileSize)
{
    int rowOffset = 0;
    int colOffset = 0;
    const int hTiles = std::ceil(input.rows / tileSize);
    const int wTiles = std::ceil(input.cols / tileSize);
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
