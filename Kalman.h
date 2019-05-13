#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

class MyKalmanFilter
{
private:
	Mat state;
    Mat measurementNoise;
    Mat processNoise;
    Mat measurement;
    Mat noisyMeasurement;
    Mat prevMeasurement;
public:
	KalmanFilter KF;
    MyKalmanFilter(Point initialPoint);
    ~MyKalmanFilter();
	void Measure(Point measurementPoint);
    Point noisyPt;
    Point estimatedPt;
    Point measuredPt;
};

