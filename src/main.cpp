#include "opencv2/highgui.hpp"
#include <iostream>
#include "number_plate_recognition.h"

int main() {
    std::string srcPath = "../../source_images/image1.jpeg";
    cv::Mat image = cv::imread(srcPath);
    if (!image.empty()) {
        cv::Mat preparedImg;
        anpr::prepare(image, preparedImg);
        std::string dstPath = "../../debug_images/image1_";
        anpr::NumberRecognition rec(preparedImg, dstPath);
        rec.detectPlate();
    }
}
