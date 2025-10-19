#include "timer.hpp"

void printTime(const std::string &msg, timer::time_point start, timer::time_point end)
{
    std::cout << msg << " --- " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
}
