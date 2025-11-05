#pragma once

#include "types.hpp"

void letterbox(forge::Image input, unsigned char *output_buffer, forge::letterbox_t &letter, int target_width, int target_height);
