#include <opencv2/opencv.hpp>
#include "opt/evaluate.h"

namespace opt {

float computePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    // Two RGB [0, 255] images are required
    assert (img1.rows == img2.rows && img1.cols == img2.cols);
    assert (img1.channels() == 3 && img2.channels() == 3);

    cv::Mat img1_norm, img2_norm;
    img1.convertTo(img1_norm, CV_32FC3, 1.0f / 255);
    img2.convertTo(img2_norm, CV_32FC3, 1.0f / 255);

    float mse = 0;
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            auto diff = img1_norm.at<cv::Vec3f>(i, j) - img2_norm.at<cv::Vec3f>(i, j);
            mse += (std::pow(diff[0], 2) + std::pow(diff[1], 2) + std::pow(diff[2], 2)) / 3.0;
        }
    }
    mse /= img1.cols * img1.rows;
    return -10 * std::log10(mse);
}


float computePSNR(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& valid_mask) {
    // Two RGB [0, 255] images are required
    // valid_mask 0: invalid pixels
    assert (img1.rows == img2.rows && img1.cols == img2.cols);
    assert (img1.rows == valid_mask.rows && img1.cols == valid_mask.cols);
    assert (img1.channels() == 3 && img2.channels() == 3 && valid_mask.channels() == 1);

    cv::Mat img1_norm, img2_norm;
    img1.convertTo(img1_norm, CV_32FC3, 1.0f / 255);
    img2.convertTo(img2_norm, CV_32FC3, 1.0f / 255);

    float mse = 0;
    bool has_value = false;
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            if (valid_mask.at<uchar>(i, j) == 0) {
                continue;
            }
            has_value = true;
            auto diff = img1_norm.at<cv::Vec3f>(i, j) - img2_norm.at<cv::Vec3f>(i, j);
            mse += (std::pow(diff[0], 2) + std::pow(diff[1], 2) + std::pow(diff[2], 2)) / 3.0;
        }
    }
    mse /= img1.cols * img1.rows;
    if (!has_value) {
        return 0;
    }
    return -10 * std::log10(mse);
}

cv::Mat visualizeDepthEdge(const cv::Mat& depth, const cv::Mat& rgb) {
    const float threshold = 0.2;

    cv::Mat dx, dy, dx2, dy2;
    cv::Sobel(depth, dx, CV_32F, 1, 0);
    cv::Sobel(depth, dy, CV_32F, 0, 1);

    cv::pow(dx, 2, dx2);
    cv::pow(dy, 2, dy2);
    dx2 = dx2 + dy2;
    cv::Mat edge;
    cv::sqrt(dx2, edge);
    // cv::Mat edge = cv::abs(dx) + cv::abs(dy);
    cv::Mat mask;
    cv::threshold(edge, mask, threshold, 1.0f, cv::THRESH_BINARY);

    // cv::Mat mask_dilated = mask;
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    // cv::dilate(mask, mask_dilated, cv::Mat());      // 3x3 kernel, iter=1
    // cv::dilate(mask, mask_dilated, kernel);      // 3x3 kernel, iter=1

    cv::Mat out = rgb.clone();
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<float>(i, j) > 0 || depth.at<float>(i, j) <= 1e-6) {
                out.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }
        }
    }
    return out;
}


} // namespace opt