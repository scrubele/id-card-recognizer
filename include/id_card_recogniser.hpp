#include "text_detector.hpp"
#include "image_preprocessor.hpp"
#include "text_recogniser.hpp"

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // Mat

namespace IDCardDetector {

    class IDCardRecogniser {
    public:
        IDCardRecogniser(int width, int height);

        virtual ~IDCardRecogniser();

        bool
        ProcessImage(cv::Mat inputImage, std::string fullName, std::string date, std::string *text, cv::Mat *image);

        void SaveResults(std::string text);

    private:

        const char *mainWindowName = "ID-CARD-RECOGNISER";
        int width, height;
        std::string imagePath;
        std::string *resultText{};

        ImagePreprocessor imagePreprocessor;
        TextDetector textDetector;
        TextRecogniser textRecogniser;

        cv::VideoCapture videoCapture;

        void SaveImage();
    };
}