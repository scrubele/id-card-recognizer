#include <../include/id_card_recogniser.hpp>
//#include "camera_runner.cpp"
// #include "utils.cpp"
#include <spdlog/spdlog.h>

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/text.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <boost/filesystem/path.hpp>
//#include <../lib/id_card_recogniser.cpp>
#include <thread>
#include <vector>
#include <iostream>
#include <thread>
#include "../include/camera.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>

#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

const char *keys =
        "{ help  h              | | Print help message. }"
        "{ input i              | | Path to input image. }"
        "{ name n               | | Full name to detect. }"
        "{ date d               | | Date to detect. }"
        "{ name n               | | Full name to detect. }"
        "{ mode m               | | 0 - process 1 file; 1 - process files from the test-data.csv; 2 - video processing }";


Camera camera;
const int frameBuffer = 50;
std::vector<cv::Mat> frameStack = *new std::vector<cv::Mat>[frameBuffer * sizeof(camera.captureVideo())];
std::vector<cv::Mat> processedStack = *new std::vector<cv::Mat>[frameBuffer * sizeof(camera.captureVideo())];
int stopSig = 0;

void processFiles();

void processFrame(std::string fullName, std::string date);

void grabFrame();


int main(int argc, char *argv[]) {

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
    int mode = parser.get<int>("mode");

    if (mode == 1) {
        processFiles();
    } else if (mode == 2) {
        cv::Mat frame;
        cv::Mat processed;
        ::frameStack.clear();
        ::processedStack.clear();

        std::thread t1(grabFrame);
        std::thread t2(processFrame, fullName, date);
        for (;;) {
            if (::processedStack.size() >= 2) {
                processed = ::processedStack.back();
                if (processed.size().width > 0)
                    cv::imshow("Processed video", processed);
            } else {
            }

            if (cv::waitKey(1) == 27) {
                std::cout << "Main: esc key is pressed by user" << std::endl;
                ::stopSig = 1;

                frameStack.clear();
                processedStack.clear();
                break;
            }
        }
        t1.join();
        t2.join();
        cv::waitKey(0);
    }

    return 0;
}


void processFrame(std::string fullName, std::string date) {
    cv::Mat frame;
    cv::Mat gauss;
    cv::Mat gray;
    cv::Mat processed;
    int i = 0;

    while (!::stopSig) {

        std::cout << "";
        std::string text;

        if (!frameStack.empty()) {

            IDCardDetector::IDCardRecogniser idCardRecogniser(1270, 720);
            cv::Mat processedImage;
            std::string recognisedText = "";
            frame = frameStack.front();


            idCardRecogniser.ProcessImage(frame, fullName, date, &recognisedText, &processedImage);
            std::cout << "Recognised text" << std::to_string(i) << " " << recognisedText << std::endl;
            frame.copyTo(processed);
            if (recognisedText.empty()) {
                recognisedText = text;
            } else {
                text = recognisedText;
            }
            cv::putText(processed, std::to_string(i), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                        cv::Scalar(255, 255, 255), 2, false);
            cv::putText(processed, recognisedText, cv::Point(100, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                        cv::Scalar(0, 255, 0), 2, false);
            if (::processedStack.size() > 2) {
                ::processedStack.pop_back();
            }
            if (!processed.empty() && ::processedStack.size() < ::frameBuffer) {

                ::processedStack.push_back(processed);
            } else if (::processedStack.size() >= ::frameBuffer) { // only in case the stack has run full...
                ::processedStack.clear();
            }
            i++;
        }

    }
    std::cout << "processFrame: esc key is pressed by user" << std::endl;
    return;
}

void grabFrame() {
    cv::Mat frame;
    frameStack.clear();
    while (!::stopSig) {
        frame = camera.captureVideo();
        if (::frameStack.size() > 2) {

            ::frameStack.pop_back();
        }

        if (::frameStack.size() < ::frameBuffer) {
            ::frameStack.push_back(frame);
        } else {
            ::frameStack.clear();
        }
    }
    std::cout << "grabFrame: esc key is pressed by user" << std::endl;
    return;
}

void processFiles(std::string fullName, std::string date) {
    std::string path = "data/test/";
    for (int i = 28; i < 32; i++) {
        std::string currentImagePath = path;
        currentImagePath = currentImagePath.append(std::to_string(i).append(".jpg"));
        std::cout << currentImagePath << std::endl;
        IDCardDetector::IDCardRecogniser idCardRecogniser(800, 600);
        std::string result;
        cv::Mat inputImage = cv::imread(currentImagePath);
        cv::Mat image;
        idCardRecogniser.ProcessImage(inputImage, fullName, date, &result, &image);
    }
}


void readFileToVectors() {
    std::ifstream file("test-data.csv");
    std::string line;

    std::vector<std::vector<std::string>> data;
    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        std::vector<std::string> record;

        while (linestream) {
            std::string s;
            if (!getline(linestream, s, ',')) break;
            record.push_back(s);
            std::cout << s << std::endl;
        }
        data.push_back(record);
    }
}

void processFiles() {
    std::ifstream file("test-data.csv");
    std::string line;
    std::ofstream resultFile("result.csv");

    std::vector<std::vector<std::string>> data;
    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        std::vector<std::string> record;

        std::string imagePath;
        std::string fullName;
        std::string date;
        getline(linestream, imagePath, ',');
        getline(linestream, fullName, ',');
        getline(linestream, date, ',');

        std::cout << imagePath << fullName << date << std::endl;
        IDCardDetector::IDCardRecogniser idCardRecogniser(800, 600);
        std::string resultString;

        cv::Mat inputImage = cv::imread(imagePath);
        cv::Mat image;
        idCardRecogniser.ProcessImage(inputImage, fullName, date, &resultString, &image);
        std::cout << resultString << std::endl;
        double scale = float(1280) / inputImage.size().height;
        resize(inputImage, inputImage, cv::Size(0, 0), scale, scale);
        cv::putText(inputImage, resultString, cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 0, 0), 2, false);

        cv::imshow("Processed video", inputImage);
        cv::waitKey(0);
        resultFile << imagePath << "\t" << resultString << "\n";
    }
    file.close();
    resultFile.close();
}
