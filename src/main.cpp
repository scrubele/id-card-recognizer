#include <../include/id_card_recogniser.hpp>
#include <spdlog/spdlog.h>

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include "opencv4/opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

//#include <../lib/id_card_recogniser.cpp>


const char *keys =
        "{ help  h              | | Print help message. }"
        "{ input i              | | Path to input image. }"
        "{ name n               | | Full name to detect. }"
        "{ date d               | | Date to detect. }";

int main(int argc, char **argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run id card detector");

    if (argc == 1 || parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string imagePath = parser.get<std::string>("input");
    std::string fullName = parser.get<std::string>("name");
    std::string date = parser.get<std::string>("date");
//    std::string fullName;
//    std::string date;

    IDCardDetector::IDCardRecogniser idCardRecogniser(800, 600);
    idCardRecogniser.ProcessImage(imagePath, fullName, date);
}