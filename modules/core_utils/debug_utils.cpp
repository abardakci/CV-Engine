#include "debug_utils.hpp"

namespace dbg
{

std::string cv_mat_flags_info(int f)
{
    static const char* depth_str[] = {
        "CV_8U","CV_8S","CV_16U","CV_16S","CV_32S","CV_32F","CV_64F","UNKNOWN"
    };
    int depth = f & ((1 << 3) - 1);
    int channels = ((f >> 3) & ((1 << 9) - 1)) + 1;
    bool cont = f & (1 << 14);
    bool sub = f & (1 << 15);

    if (depth < 0 || depth > 6)
        depth = 7; // out-of-range -> "UNKNOWN"

    char buf[128];
    std::snprintf(buf, sizeof(buf), "%sC%d (cont=%d, sub=%d)",
                  depth_str[depth], channels, cont, sub);
    return std::string(buf);
}

}