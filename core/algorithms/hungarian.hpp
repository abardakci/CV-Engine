#include <opencv2/opencv.hpp>
#include "types.hpp"
#include "postprocess.hpp"

std::vector<int> hungarian(cv::Mat& cost_matrix);
cv::Mat computeCostMatrix(const std::vector<Track>& tracks, const std::vector<Box>& detects);
