#include "file_io.hpp"

namespace forge
{
    std::vector<char> load_binary_file(const std::string &path)
    {
        std::ifstream reader(path, std::ios::binary | std::ios::ate);
        if (!reader)
        {
            std::cerr << "File not open error: " << path << std::endl;
        }

        size_t size = reader.tellg();
        if (size <= 0)
        {
            std::cerr << "File size invalid error: " << path << std::endl;
        }

        std::vector<char> binary;
        binary.resize(size);

        reader.seekg(0, std::ios::beg);
        if (!reader.read(binary.data(), size))
        {
            std::cerr << "File read error: " << path << std::endl;
        }

        return binary;
    }
}
