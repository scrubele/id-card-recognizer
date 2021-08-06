#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // Mat
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <spdlog/spdlog.h>

namespace IDCardDetector {

    class ImagePreprocessor {
    public:
        explicit ImagePreprocessor(char const *);

        bool PreprocessImage(cv::Mat, cv::Mat *);

        bool MakeImageMask(cv::Mat inputImage, cv::Mat *outputImage);

        bool ProcessImage(cv::Mat inputImage, cv::Mat *outputImage, cv::Mat *thresholdOutput);

    private:
        const char *windowName;
        cv::Mat image;
    };
}
