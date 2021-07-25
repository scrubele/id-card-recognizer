#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // Mat
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <spdlog/spdlog.h>
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

namespace IDCardDetector {

    class TextDetector {
    public:
        explicit TextDetector(char const *);

        bool ExtractText(cv::Mat inputImage, std::string *text);

        bool RecogniseText(cv::Mat inputImage, std::string *text);

        bool ProcessImage(cv::Mat inputImage, std::string *text);

    private:
        const char *windowName;
        cv::Mat image;
        tesseract::TessBaseAPI *ocr;
    };
}
