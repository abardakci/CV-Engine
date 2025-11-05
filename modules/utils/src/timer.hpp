#pragma once

#include <chrono>
#include <iostream>
#include <format>
#include <string>

namespace forge
{
    inline void print_time(std::string_view msg, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
    {
        uint64_t count = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << std::format("[TIMER] {} - {}ms\n", msg, count);
    }

    inline auto now()
    {
        return std::chrono::steady_clock::now();
    }
    
}
