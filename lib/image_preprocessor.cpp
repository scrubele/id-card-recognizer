
#include "image_preprocessor.hpp"

#include <iostream>

using namespace std;


namespace IDCardDetector {


    ImagePreprocessor::ImagePreprocessor(char const *_windowName)
            : windowName{_windowName} {
    }


    bool ImagePreprocessor::PreprocessImage(cv::Mat inputImage, cv::Mat *outputImage) {
        cv::Mat grayImage;
        inputImage.copyTo(grayImage);

        cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(13, 5));
        cv::Mat sqKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(21, 21));

        // Smooth image
        cv::GaussianBlur(grayImage, grayImage, cv::Point(3, 3), 0);
        cv::Mat blackhat;
        cv::morphologyEx(grayImage, blackhat, cv::MORPH_BLACKHAT, rectKernel);

        cv::Mat gradX;
        cv::Sobel(blackhat, gradX, CV_32F, 1, 0, 1);
        gradX = cv::abs(gradX);
        double minVal, maxVal;
        cv::minMaxLoc(gradX, &minVal, &maxVal);
        gradX.convertTo(gradX, CV_8U);
        gradX = ((uint8_t) 255 * ((gradX - (uint8_t) minVal) / ((uint8_t) maxVal - (uint8_t) minVal)));

        cv::morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectKernel);

        // Apply Otsu's thresholding method
        cv::Mat threshold;
        cv::threshold(gradX, threshold, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        // Another closing operation using sqKernel to close gaps between lines of MRZ
        cv::morphologyEx(threshold, threshold, cv::MORPH_CLOSE, sqKernel);

        // Perform series of erosions to break apart connected components
        cv::dilate(threshold, threshold, cv::noArray(), cv::Point(-1, -1), 4);
        cv::GaussianBlur(threshold, threshold, cv::Point(7, 7), 0);
        cv::imshow("threshold", threshold);
        *outputImage = threshold;
    }

    bool ImagePreprocessor::MakeImageMask(cv::Mat inputImage, cv::Mat *outputImage) {
        cv::Mat mask = cv::Mat(inputImage.rows, inputImage.cols, CV_64F, cvScalar(0.));

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(inputImage, contours, hierarchy, cv::RETR_TREE,
                         cv::CHAIN_APPROX_SIMPLE);
//        std::cout << "contours" << contours.size() << std::endl;
        for (const auto &contour : contours) {
            cv::Rect box = cv::boundingRect(contour);
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1,
                             cv::Scalar(255, 255, 255), -1);
//            cv::imshow("mask1", mask);
//            cv::waitKey(0);
        };
        cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(11, 11));
        cv::dilate(mask, mask, rectKernel, cv::Point(-1, -1), 1);

        cv::imshow("mask", mask);

        cv::Mat convertedMask;
        mask.convertTo(convertedMask, CV_8U);
        *outputImage = convertedMask;

    }


    bool ImagePreprocessor::ProcessImage(cv::Mat inputImage, cv::Mat *outputImage) {
        cv::Mat thresholdImage;
        PreprocessImage(inputImage, &thresholdImage);
        cv::Mat maskImage;
        MakeImageMask(thresholdImage, &maskImage);
        cv::Mat textOnlyImage;
        cv::bitwise_and(inputImage, inputImage, textOnlyImage, maskImage);
        cv::imshow("bitwise_and", textOnlyImage);
        *outputImage = textOnlyImage;
    }


}