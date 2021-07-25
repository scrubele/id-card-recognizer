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

        bool ProcessImage(std::string imagePath, std::string fullName, std::string date);

    private:

        const char *mainWindowName = "ID-CARD-RECOGNISER";
        int width, height;
        std::string imagePath;

        // Holds the results of id card recogniser
        std::string *resultText{};

        ImagePreprocessor imagePreprocessor;
        TextDetector textDetector;
        TextRecogniser textRecogniser;

        void SaveImage();
    };
}