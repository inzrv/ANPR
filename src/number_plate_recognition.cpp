//
// Created by Иван Назаров on 25.01.2023.
//
#include "number_plate_recognition.h"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <iostream>

anpr::NumberRecognition::NumberRecognition(const cv::Mat& image, const std::string& debugPath)
: sourceImg(image.clone())
, platePos()
, numberStr()
, debugPath(debugPath)
{
    writeToDebug(sourceImg, "source.png");
}

void anpr::NumberRecognition::detectPlate() {

    cv::Mat sourceCopyImg = sourceImg.clone();

    // detect edges
    cv::Mat edgesImg(sourceCopyImg.size(), CV_8UC1);
    detectEdges(sourceCopyImg, edgesImg);
    writeToDebug(edgesImg, "edges.png");

    //dilation
    cv::Mat dilatationImg(edgesImg.size(), CV_8UC1);
    dilation(edgesImg, dilatationImg);
    writeToDebug(dilatationImg, "dilatation.png");

    // find and draw contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours( dilatationImg, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::Mat contoursImg = cv::Mat::zeros(dilatationImg.size(), CV_8UC3);
    drawContours(contoursImg, contours);
    writeToDebug(contoursImg, "contours.png");

    // sort and approximate contours
    sortContours(contours);
    std::vector<std::vector<cv::Point>> approxContours;
    approximateContours(contours, approxContours);
    cv::Mat approxContoursImg = cv::Mat::zeros(contoursImg.size(), CV_8UC3);
    drawContours(approxContoursImg, approxContours);
    writeToDebug(approxContoursImg, "approx_contours.png");

    // find good contours
    std::vector<std::vector<cv::Point>> goodContours;
    for (int i = 0; i < approxContours.size(); ++i) {
        if( isGoodContour(approxContours[i])) {
            goodContours.push_back(approxContours[i]);
        }
    }

    // sort good contours
    sortContours(goodContours);
    cv::Mat goodContourImg = cv::Mat::zeros(approxContoursImg.size(), CV_8UC1);
    for (int i = 0; i < goodContours.size(); ++i) {
        cv::polylines(goodContourImg, goodContours[i], true, 255);
    }
    writeToDebug(goodContourImg, "good_contours.png");

    goodContours.resize(1);

    // build bounding rotated rectangle
    cv::Mat boundingRectsImg(sourceImg);
    cv::cvtColor(boundingRectsImg, boundingRectsImg, cv::COLOR_GRAY2BGR);
    for (auto contour: goodContours) {
        cv::RotatedRect boundingRect = cv::minAreaRect(contour);
        drawRotRect(boundingRectsImg, boundingRect);
    }
    writeToDebug(boundingRectsImg, "plate.png");
}

void anpr::NumberRecognition::writeToDebug(const cv::Mat& image, const std::string& name) {
    cv::imwrite(debugPath + name, image);
}

void anpr::detectEdges(const cv::Mat& src, cv::Mat& dst) {
    double threshold1 = 200;
    double threshold2 = 30;
    cv::Canny(src, dst, threshold1, threshold2);
}

void anpr::dilation(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat element = getStructuringElement( cv::MORPH_CROSS,{2, 1});
    cv::dilate( src, dst, element);
}

void anpr::sortContours(std::vector<std::vector<cv::Point>> &contours) {
    std::sort(contours.begin(), contours.end(),
              [](const auto& c1, const auto& c2)
              {
                  return cv::contourArea(c1) > cv::contourArea(c2);
              }
    );
}

void anpr::approximateContours(
                            std::vector<std::vector<cv::Point>>& contours,
                            std::vector<std::vector<cv::Point>>& approxContours
                            ) {
    double epsilon = 15;
    bool closed = true;
    for (auto& contour : contours) {
        std::vector<cv::Point> approxContour;
        cv::approxPolyDP( cv::Mat(contour), approxContour, epsilon, closed );
        if (approxContour.size() >= 4 && approxContour.size() <= 7) {
            approxContours.push_back(approxContour);
        }
    }
}

void anpr::drawContours(cv::Mat& contoursImg, const std::vector<std::vector<cv::Point>>& contours) {
    for(const auto& contour : contours) {
        cv::Scalar color( rand()&255, rand()&255, rand()&255 );
        cv::polylines(contoursImg, contour, true, color, 1);
    }
}

void anpr::drawRotRect(cv::Mat& imgRects, const cv::RotatedRect& rect) {
    cv::Scalar color( 255, 0, 0 );
    cv::Point2f points[4];
    rect.points(points);
    for (int j = 0; j < 4; ++j) {
        cv::line( imgRects, points[j], points[(j + 1) % 4], color, 2);
    }
}

void anpr::shiftContour(std::vector<cv::Point>& contour, cv::Point shift) {
    for(auto& point : contour) {
        point -= shift;
    }
}

int anpr::numberOfNeighbours(cv::Mat& img, cv::Point pixel) {
    int xMin = (pixel.x > 0) ? pixel.x - 1 : 0;
    int yMin = (pixel.y > 0) ? pixel.y - 1 : 0;
    int xMax = (pixel.x < img.cols - 1) ? pixel.x + 1 : img.cols - 1;
    int yMax = (pixel.y < img.rows - 1) ? pixel.y + 1 : img.rows - 1;
    int sum = 0;
    for (int y = yMin; y <= yMax; ++y) {
        for (int x = xMin; x <= xMax; ++x) {
            if (img.at<uchar>(y,x) == 255) {
                sum++;
            }
        }
    }
    return sum;
}

bool anpr::isGoodContour(std::vector<cv::Point> contour) {
    cv::Rect boundingRect = cv::boundingRect(contour);
    shiftContour(contour, boundingRect.tl());
    cv::Mat contourImg = cv::Mat::zeros(boundingRect.size(), CV_8UC1);
    cv::polylines(contourImg, contour, true, 255);
    for (int row = 0; row < contourImg.rows; ++row) {
        for (int col = 0; col < contourImg.cols; ++col) {
            if( contourImg.at<uchar>(row,col) != 0) {
                if (numberOfNeighbours(contourImg, {col, row}) > 4) {
                    return false;
                }
            }
        }
    }
    return true;
}

void anpr::prepare(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat resizedImg;
    cv::resize(src, resizedImg, {512, 512});
    cv::Mat grayImg;
    if (resizedImg.channels() == 3) {
        cv::cvtColor(resizedImg, grayImg, cv::COLOR_BGR2GRAY);
    } else {
        grayImg = resizedImg;
    }
    cv::bilateralFilter(grayImg, dst, 3,17, 17);
}

std::vector<cv::Point>& anpr::getBestContour(std::vector<std::vector<cv::Point>>& contours) {
    return contours[0];
}