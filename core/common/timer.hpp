#include <chrono>
#include <iostream>

using timer = std::chrono::steady_clock;
void printTime(const std::string &msg, timer::time_point start, timer::time_point end);
