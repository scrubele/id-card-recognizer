#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // Mat
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <spdlog/spdlog.h>
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>
#include <difflib/difflib.h>

namespace IDCardDetector {

    class TextRecogniser {
    public:
        explicit TextRecogniser(char const *);

        int RecogniseName(std::string inputText, std::string name, std::string *recognisedValues);

        bool ProcessText(std::string inputText, std::string fullName, std::string date,std::map<std::string, int> *recognisedValues);

    private:
    };
}
