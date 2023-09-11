// Wrapper class for OpenCV Mat with some additional meta data
// It support some image processing functions for the optimization
#pragma once
#include <opencv2/opencv.hpp>
#include <mLibCore.h>
#include <filesystem>

namespace opt {

class Image {
public:
    Image() {}

    Image(const int id, const std::string& path, int mode = cv::IMREAD_UNCHANGED): id_(id), image_path(path) {
        // Read from file
        image_data = cv::imread(path, mode);
        assert (channels() == 1 || channels() == 3);
    }

    Image(const Image& image) {
        // Copy constructor
        image_data = image.image_data.clone();
        image_path = image.image_path;
        id_ = image.id_;
    }

    Image(const int id, const int height, const int width, const int channels): id_(id), image_path("") {
        // Initialize empty image Mat
        if (channels == 1) image_data = cv::Mat(height, width, CV_32F);
        else if (channels == 3) image_data = cv::Mat(height, width, CV_32FC3);
        else throw MLIB_EXCEPTION("invalid channel number");
    }

    Image(const int id, const cv::Mat& mat): id_(id), image_path("") {
        image_data = mat.clone();
        assert (channels() == 1 || channels() == 3);
    }

    ~Image() {}

    bool inRange(const float u, const float v) {
        return (u >= 0 && u <= width()-1 && v >= 0 && v <= height()-1);
    }

    bool inRange(int u, int v) {
        return (u >= 0 && u <= width()-1 && v >= 0 && v <= height()-1);
    }

    void convertToGrayscale() {
        assert (channels() == 3);
        cv::Mat gray;
        cv::cvtColor(image_data, gray, cv::COLOR_BGR2GRAY);
        image_data = gray;
        assert (channels() == 1);
    }

    void normalize() {
        assert (channels() == 1);
        cv::Mat new_image;
        image_data.convertTo(new_image, CV_32F, 1.0/255);
        image_data = new_image;
    }

    void applyGaussian(int kernel_size, float sigma) {
        cv::Mat new_image;
        cv::GaussianBlur(image_data, new_image, cv::Size(kernel_size, kernel_size), sigma);
        image_data = new_image;
    }

    void computeGradientX() {
        assert (image_data.channels() == 1);
        cv::Mat gradientX;
        cv::Sobel(image_data, gradientX, CV_32F, 1, 0, 3);
        image_data = gradientX;
    }

    void computeGradientY() {
        assert (image_data.channels() == 1);
        cv::Mat gradientY;
        cv::Sobel(image_data, gradientY, CV_32F, 0, 1, 3);
        image_data = gradientY;
    }

    void computeDepthBoundaryMask(const float threshold) {
        // compute the [0, 1] mask of the depth boundary with in-place modification
        cv::Mat gradientX, gradientY;
        cv::Sobel(image_data, gradientX, CV_32F, 1, 0, 3);
        cv::Sobel(image_data, gradientY, CV_32F, 0, 1, 3);
        cv::Mat mask(height(), width(), CV_32F);

        for (int y = 0; y < height(); y += 1) {
            for (int x = 0; x < width(); x += 1) {
                float dx = gradientX.at<float>(y, x);
                float dy = gradientY.at<float>(y, x);
                float grad = std::sqrt(dx * dx + dy * dy);
                if (grad <= threshold)
                    mask.at<float>(y, x) = 1.0f;
                else
                    mask.at<float>(y, x) = 0;
            }
        }
        image_data = mask;
    }

    void resize(int target_height, int target_width) {
        cv::Mat new_image;
        cv::resize(image_data, new_image, cv::Size(target_width, target_height), cv::INTER_LINEAR);
        image_data = new_image;

        // // Update the metadata
        // float scale_x = (float) target_height / width();
        // float scale_y = (float) target_width / height();
    }

    template <typename T>
    T getPixelValue(const float u, const float v) {
        // Bilinear interpolation
        assert (inRange(u, v));
        assert (image_data.channels() == 1);    // "Support only one-channel images

        int u0 = std::floor(u);
        int u1 = std::ceil(u);
        int v0 = std::floor(v);
        int v1 = std::ceil(v);

        assert (inRange(u0, v0));
        assert (inRange(u1, v1));
        T out = (u1 - u) * (v1 - v) * image_data.at<T>(v0, u0)
            + (u - u0) * (v1 - v) * image_data.at<T>(v0, u1)
            + (u1 - u) * (v - v0) * image_data.at<T>(v1, u0)
            + (u - u0) * (v - v0) * image_data.at<T>(v1, u1);
        return out;
    }

    template <typename T>
    T getNearestPixelValue(const float u, const float v) {
        int u0 = int(round(u));
        int v0 = int(round(v));
        // std::cout << u0 << " " << v0 << " " << meta.height << " " << meta.width << std::endl;
        assert (inRange(u0, v0));
        return image_data.at<T>(v0, u0);
    }

    ml::vec3f getRGBPixelValue(const float u, const float v) {
        // Expect the image_data in RGB format (3 channels + 0-255 range)
        // Output vec3f with 0-1
        assert (inRange(u, v));
        assert (channels() == 3);    // "Support only 3-channel images
        int u0 = std::floor(u);
        int u1 = std::ceil(u);
        int v0 = std::floor(v);
        int v1 = std::ceil(v);

        assert (inRange(u0, v0));
        assert (inRange(u1, v1));

        ml::vec3f out;
        // Bilinear interpolation
        out.r = (u1 - u) * (v1 - v) * (float)image_data.at<cv::Vec3b>(v0, u0)[2]
            + (u - u0) * (v1 - v) * (float)image_data.at<cv::Vec3b>(v0, u1)[2]
            + (u1 - u) * (v - v0) * (float)image_data.at<cv::Vec3b>(v1, u0)[2]
            + (u - u0) * (v - v0) * (float)image_data.at<cv::Vec3b>(v1, u1)[2];

        out.g = (u1 - u) * (v1 - v) * (float)image_data.at<cv::Vec3b>(v0, u0)[1]
            + (u - u0) * (v1 - v) * (float)image_data.at<cv::Vec3b>(v0, u1)[1]
            + (u1 - u) * (v - v0) * (float)image_data.at<cv::Vec3b>(v1, u0)[1]
            + (u - u0) * (v - v0) * (float)image_data.at<cv::Vec3b>(v1, u1)[1];

        out.b = (u1 - u) * (v1 - v) * (float)image_data.at<cv::Vec3b>(v0, u0)[0]
            + (u - u0) * (v1 - v) * (float)image_data.at<cv::Vec3b>(v0, u1)[0]
            + (u1 - u) * (v - v0) * (float)image_data.at<cv::Vec3b>(v1, u0)[0]
            + (u - u0) * (v - v0) * (float)image_data.at<cv::Vec3b>(v1, u1)[0];

        out /= 255.f;
        return out;
    }

    cv::Mat& getCVImage() { return image_data; }

    int height() { return image_data.rows; }
    int width() { return image_data.cols; }
    int channels() { return image_data.channels(); }

    std::string getImageName() const {
        std::filesystem::path p(image_path);
        return p.stem();
    }
    int getId() const { return id_; }

    std::string image_path;
private:
    int id_;
    cv::Mat image_data;
};


} // namespace opt
