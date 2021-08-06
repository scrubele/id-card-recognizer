#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

//using namespace cv;
//using namespace std;

class Camera
{
public:
    Camera(void);
    ~Camera(void);
    cv::Mat captureVideo(void);

private:
    cv::Mat frame;
    double dWidth;
    double dHeight;
    double fps;
    cv::VideoCapture cap;


};


