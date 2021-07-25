#include "text_detector.hpp"
#include "image_preprocessor.hpp"

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // Mat

namespace IDCardDetector {

    class IDCardRecogniser {
    public:
        IDCardRecogniser(std::string imagePath, int width, int height);

        virtual ~IDCardRecogniser();

        bool ProcessImage();

    private:

        const char *mainWindowName = "ID-CARD-RECOGNISER";
        int width, height;
        std::string imagePath;

        // Holds the results of id card recogniser
        std::string *resultText{};

        ImagePreprocessor imagePreprocessor;
        TextDetector textDetector;

        void SaveImage();
    };
}