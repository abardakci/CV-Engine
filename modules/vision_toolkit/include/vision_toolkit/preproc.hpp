#pragma once

#include "utils/types.hpp"

void letterbox(ImageWrapper input, unsigned char *output_buffer, letterbox_t &letter, int target_width, int target_height);
