
#include "text_detector.hpp"

#include <iostream>


namespace IDCardDetector {


    TextDetector::TextDetector(char const *_windowName)
            : windowName{_windowName} {
        this->ocr = new tesseract::TessBaseAPI();
        this->ocr->SetPageSegMode(tesseract::PSM_AUTO);
    }

    bool TextDetector::ExtractText(cv::Mat inputImage, std::string *text) {
        this->ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
        this->ocr->SetImage(inputImage.data, inputImage.cols, inputImage.rows, 3, inputImage.step);
        std::string outText = std::string(this->ocr->GetUTF8Text());
//        std::cout << "outText:"<<outText<<stdL::endl;
        this->ocr->End();
        *text = outText;
    }


    bool TextDetector::RecogniseText(cv::Mat inputImage, std::string *text) {
        std::string extractedText = "";
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(inputImage, contours, hierarchy, cv::RETR_TREE,
                         cv::CHAIN_APPROX_SIMPLE);
        std::cout << "contours" << contours.size() << std::endl;
        for (const auto &contour : contours) {
            cv::Rect box = cv::boundingRect(contour);
            std::cout << box.x << " " << box.y << " " << box.width << " " << box.height << std::endl;
            // check the box within the image plane
            if (0 <= box.x
                && 0 <= box.width
                && box.x + box.width <= inputImage.cols
                && 0 <= box.y
                && 0 <= box.height
                && box.y + box.height <= inputImage.rows) {

                cv::Rect croppedROI(box.x, box.y, box.width, box.height);
                cv::Mat croppedImage = inputImage(croppedROI);
                std::string croppedText;
                ExtractText(croppedImage, &croppedText);
                std::cout << croppedText << std::endl;
                cv::imshow("cropped", croppedImage);
                cv::waitKey(0);

                extractedText += croppedText;
            } else {
                std::cout << "box out of image plane " << std::endl;
                // box out of image plane, do something...
            }

        };
        *text = extractedText;
    }


    bool TextDetector::ProcessImage(cv::Mat inputImage, std::string *text) {
        std::string extractedText;
        RecogniseText(inputImage, &extractedText);
        std::cout << extractedText << std::endl;
    }


}