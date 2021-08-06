#include <unistd.h>
#include "camera.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include "opencv4/opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <boost/filesystem/path.hpp>

using namespace std;

//VideoCapture cap(0);

Camera::Camera(void) {
    cout << " Camera warming up..." << endl;
    // int deviceID = 0;  
    if (!this-> cap.open(cv::CAP_V4L)) {
        cout << "Cannot open the video cam" << endl;
    usleep(10);

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 10);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 2);
    // cap.set(cv::CAP_PROP_FOURCC,VideoWriter_fourcc('M','J','P','G'));
    dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps = cap.get(cv::CAP_PROP_FPS);

    cout << "Frame size : " << dWidth << " x " << dHeight << " --- fps: " << fps << endl;

    cap >> frame;
}


Camera::~Camera(void) {
    cout << "Shutting down camera and closing files..." << endl;
    cap.release();
}


cv::Mat Camera::captureVideo(void) {
    cap >> frame;
    //cout << "In VideoCapture Height = " << frame.rows << " .. Width = " << frame.cols << endl;
    return frame;
}