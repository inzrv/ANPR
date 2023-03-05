//
// Created by Иван Назаров on 25.01.2023.
//
#pragma once
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <string>

namespace anpr {
    class NumberRecognition {
    public:
        explicit NumberRecognition(const cv::Mat &image, const std::string &debugPath = "../debug_images/");

        void detectPlate();

    private:
        cv::Mat sourceImg;
        cv::RotatedRect platePos;
        std::string numberStr;
        const std::string debugPath;

        void writeToDebug(const cv::Mat &image, const std::string &name);
    };

    void sortContours(std::vector<std::vector<cv::Point>> &contours);

    void approximateContours(
            std::vector<std::vector<cv::Point>> &contours,
            std::vector<std::vector<cv::Point>> &approxContours);

    void detectEdges(const cv::Mat &src, cv::Mat &dst);

    void dilation(const cv::Mat &src, cv::Mat &dst);

    void drawContours(cv::Mat &contoursImg, const std::vector<std::vector<cv::Point>> &contours);

    void drawRotRect(cv::Mat& imgRects, const cv::RotatedRect& rect);

    void shiftContour(std::vector<cv::Point> &contour, cv::Point shift);

    bool isGoodContour(std::vector<cv::Point> contour);

    int numberOfNeighbours(cv::Mat &img, cv::Point pixel);

    void prepare(const cv::Mat& src, cv::Mat& dst);

    std::vector<cv::Point>& getBestContour(std::vector<std::vector<cv::Point>>& contours);
}