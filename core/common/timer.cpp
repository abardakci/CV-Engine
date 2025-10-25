#include "timer.hpp"

void print_time(const std::string &msg, timer::time_point start, timer::time_point end)
{
    uint64_t count = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << std::format("[TIMER] {} | {}ms", msg, count);
}
