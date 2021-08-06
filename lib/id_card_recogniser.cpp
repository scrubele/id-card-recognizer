#include "id_card_recogniser.hpp"

#include <utility>
#include <spdlog/spdlog.h>
#include <fstream>
#include <boost/filesystem.hpp>

namespace IDCardDetector {
    IDCardRecogniser::IDCardRecogniser(int width, int height)
            : textDetector{mainWindowName}, imagePreprocessor{mainWindowName}, textRecogniser{mainWindowName} {
        this->width = width;
        this->height = height;
    }

    void IDCardRecogniser::SaveResults(std::string text) {
        std::ofstream resultFile;
        resultFile.open("result.txt", std::ios_base::app);
        resultFile << text;
    }

    bool IDCardRecogniser::ProcessImage(cv::Mat inputImage, std::string fullName, std::string date,
                                        std::string *text, cv::Mat *image) {

        std::cout << inputImage.size() << std::endl;
        // cv::imshow("inputImage", inputImage);
        // imshow("Original Image", inputImage);
        cv::Mat resizedImage;
        std::cout << inputImage.size();
        double scale = float(1280) / inputImage.size().width;
        resize(inputImage, resizedImage, cv::Size(0, 0), scale, scale);
        cv::Mat grayImage;
        cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);

        cv::Mat processedImage, thresholdImage;
        imagePreprocessor.ProcessImage(grayImage, &processedImage, &thresholdImage);
        *image = processedImage;
        // cv::imshow("processedImage", processedImage);
        std::string extractedText;
        textDetector.ProcessImage(processedImage, thresholdImage, &extractedText);

        std::cout << extractedText << std::endl;
        std::map<std::string, int> currentRecognised;
        textRecogniser.ProcessText(extractedText, fullName, date, &currentRecognised);
        std::string resultText = "";
        for (auto itr = currentRecognised.begin(); itr != currentRecognised.end(); ++itr) {
            resultText.append(itr->first);
            resultText.append(" ");
            resultText.append(std::to_string(itr->second));
            resultText.append(" ");
            std::cout << itr->first << '\t' << itr->second << '\n';
        }
        *text = resultText;
    }


    void IDCardRecogniser::SaveImage() {

    }

    IDCardRecogniser::~IDCardRecogniser() = default;
}
