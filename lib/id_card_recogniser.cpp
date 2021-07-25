#include "id_card_recogniser.hpp"

#include <utility>
#include <spdlog/spdlog.h>


namespace IDCardDetector {
    IDCardRecogniser::IDCardRecogniser(std::string imagePath, int width, int height)
            : textDetector{mainWindowName}, imagePreprocessor{mainWindowName} {
        this->imagePath = imagePath;
        this->width = width;
        this->height = height;
    }

    bool IDCardRecogniser::ProcessImage() {
        std::cout << "image path: " << this->imagePath << std::endl;

        cv::Mat inputImage = cv::imread(this->imagePath);
//        imshow("Original Image", inputImage);

        cv::Mat resizedImage;
        resize(inputImage, resizedImage, cv::Size(), 0.25, 0.25);
        cv::Mat grayImage;
        cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);

        cv::Mat processedImage;
        imagePreprocessor.ProcessImage(grayImage, &processedImage);
        std::string *extractedText;
        textDetector.ProcessImage(processedImage, extractedText);
//        cv::imwrite( "test.jpg", grayImage);
        cv::waitKey(0);
        cv::waitKey(1);
    }


    void IDCardRecogniser::SaveImage() {

    }

    IDCardRecogniser::~IDCardRecogniser() = default;
}
