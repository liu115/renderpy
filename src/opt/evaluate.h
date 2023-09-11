#pragma once

namespace opt {
    cv::Mat visualizeDepthEdge(const cv::Mat& depth, const cv::Mat& rgb);
    float computePSNR(const cv::Mat& img1, const cv::Mat& img2);
    float computePSNR(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& valid_mask);

} // namespace opt