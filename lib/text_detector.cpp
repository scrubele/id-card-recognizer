
#include "text_detector.hpp"

#include <iostream>

namespace IDCardDetector {


    TextDetector::TextDetector(char const *_windowName)
            : windowName{_windowName} {
        this->ocr = new tesseract::TessBaseAPI();
        this->ocr->SetPageSegMode(tesseract::PSM_AUTO);
    }

    bool TextDetector::RecogniseText(cv::Mat inputImage, std::string *text) {
        this->ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
        this->ocr->SetVariable("user_defined_dpi", "120");
        this->ocr->SetImage(inputImage.data, inputImage.cols, inputImage.rows, 3, inputImage.step);
        std::string outText = std::string(this->ocr->GetUTF8Text());
        this->ocr->End();
        *text = outText;
    }

    bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
        double i = fabs(contourArea(cv::Mat(contour1)));
        double j = fabs(contourArea(cv::Mat(contour2)));
        return (i > j);
    }

    bool TextDetector::ExtractText(cv::Mat inputImage, cv::Mat thresholdInput, std::string *text) {
        std::string extractedText = "";
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(inputImage, contours, hierarchy, cv::RETR_TREE,
                         cv::CHAIN_APPROX_SIMPLE);
        std::sort(contours.begin(), contours.end(), compareContourAreas);
        std::cout << "contours" << contours.size() << std::endl;
        int i = 0;
        for (const auto &contour : contours) {
            double area = cv::contourArea(contour);

            cv::Rect box = cv::boundingRect(contour);

            if (area >= 3000.0) {
                if (0 <= box.x
                    && 0 <= box.width
                    && box.x + box.width <= inputImage.cols
                    && 0 <= box.y
                    && 0 <= box.height
                    && box.y + box.height <= inputImage.rows) {

                    cv::Rect croppedROI(box.x, box.y, box.width, box.height);
                    cv::Mat croppedImage = inputImage(croppedROI);
                    cv::Mat colorfulCropped;
                    cv::equalizeHist(croppedImage, croppedImage);
                    cv::threshold(croppedImage, croppedImage, 40, 255, cv::THRESH_BINARY);

                    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(3, 1));
                    cv::erode(croppedImage, croppedImage, rectKernel, cv::Point(-1, -1));


                    cv::cvtColor(croppedImage, colorfulCropped, cv::COLOR_BGR2RGB);
                    std::string croppedText;
                    RecogniseText(colorfulCropped, &croppedText);
                    std::cout << croppedText << std::endl;
                    extractedText += croppedText;

                    croppedImage.release();
                    colorfulCropped.release();
                } else {
                    std::cout << "box out of image plane " << std::endl;
                }
            } else {
                break;
            }
            i += 1;

        };
        *text = extractedText;

        contours.clear();
        hierarchy.clear();
    }


    bool isNotAlnum(char c) {
        return isalnum(c) == 0;
    }

    bool TextDetector::ProcessImage(cv::Mat inputImage, cv::Mat thresholdInput, std::string *text) {
        std::string extractedText;
        ExtractText(inputImage, thresholdInput, &extractedText);
        extractedText.erase(std::remove(extractedText.begin(), extractedText.end(), '\n'),
                            extractedText.end());
        std::cout << extractedText << std::endl;
        extractedText.erase(std::remove_if(extractedText.begin(), extractedText.end(), isNotAlnum),
                            extractedText.end());
        *text = extractedText;
    }

}