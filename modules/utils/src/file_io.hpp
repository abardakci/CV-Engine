#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace forge
{
    std::vector<char> load_binary_file(const std::string &path);
}
