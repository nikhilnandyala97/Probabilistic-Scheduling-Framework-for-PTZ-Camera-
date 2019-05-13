#ifndef _MOTION_H
#define _MOTION_H

#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp" 
#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include "medianFilter.h"
#include "curl/curl.h"
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <math.h>
#include <ctime>
#include <time.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <algorithm>
#include <cstdlib>
#include <stdlib.h> 


using namespace std;
using namespace cv;
using namespace dnn;

Rect FindContourAndGetMaxRect2(Mat frame);

int CalTimeDifference(struct timeval tm1, struct timeval tm2);

Mat FaceDetectByCascade(Mat frame, Mat mask, CascadeClassifier *face_cascade);

Mat BackGroundDetection(Mat frame, Mat mask,Ptr<BackgroundSubtractor> pMog);

Mat PeopleDetectByHOG( Mat frame , Mat mask, HOGDescriptor hog);

Rect Rect_AdjustSizeAroundCenter(Rect rect, double width_factor, double height_factor,
	  int videoWidth = 0, int videoHeight = 0);


Rect Rect_AdjustRelativePosition(Rect mediumRect, Rect smallRect);

Rect Rect_WithCenter(Point center, Size orginalRect);

Point Rect_GetCenter(Rect rectangle);

Point Point_AdjustRelativePosition(Rect mediumRect, Point point);

template <typename T> string NumberToString ( T Number );

bool IsIntegerBetween(int number, int low, int hight);

Rect SupplementRect(Rect rect, int videoWidth, int videoHeight);

int GetMedianResult(vector<int> *v ,int insert, int window);

Rect GetMedianRectResult(vector<int> *x, vector<int> *y, Rect rect , int window);

void mergeSort(int *s, int m, int n);

void merge(int *s, int m, int k, int n);

int CalculateInterpolation(float interpolant, int a, int b);

Rect Zoom_CubicSpline(Rect initialRect, Rect lastRect, 
  int startFrame, int endFrame, int currentFrame);

Rect ChangeRatioOfRectCut(Rect zoomRect, Rect targetRect);

Rect ChangeRatioOfRect(Rect zoomRect, float targetRectRatio);

double TwoDimensionalGaussianfunction(double A, double centerX, double centerY, 
  double eX, double eY, double currentX, double currentY);

void CalculateGasussianROI(Mat_<float> penalty, Rect maxRect, double amplitude);

void MultiplyFloatInRect(Mat_<float> frame, float value, Rect rect);

void SubstractByFloatInRect(Mat_<float> frame, float value, Rect rect);

int SumAllPixelsInRect(Mat frame, Rect rect);

int SumAllPixelsInVector1d(Mat mat_original, vector<Point> vector);
string GetCurrentDateTime();

bool IsBetweenTwoInt(int value,int left, int right);
Mat FaceDetectorDNN(Mat frame, Mat mask, dnn::Net net );
Mat HumanDetectionYOLO( Mat frame , Mat mask, Net onet);
vector<String> getOutputsNames(const Net& net);
vector <Rect> BackGroundDetectionInit(Mat frame, Mat mask,Ptr<BackgroundSubtractor> pMog );
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata);
Mat curlImg(const char *img_url, int timeout);
void curlptzcam(const char *img_url , int to);
int nextStateGenerator(int A[]);
long double SumofPixels(Mat mask );

extern int accumulate_time;
extern Size video_size;
extern int mask_add_step;
extern double BG_scale_factor;
extern double Body_scale_factor;
extern double Face_scale_factor;
extern int display_time;
extern double time_zoomIn_percent;
extern double time_stay_percent;
extern int running_time[];
extern int pause_frame;
 const size_t inWidth = 300;
 const size_t inHeight = 300;
 const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);
const int oWidth=416;
const int oHeight=416;
const float confThreshold = 0.5;
const float nmsThreshold = 0.4;



#endif
