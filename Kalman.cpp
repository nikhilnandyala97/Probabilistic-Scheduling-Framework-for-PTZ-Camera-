//#pragma once
#include "Kalman.h"

MyKalmanFilter::MyKalmanFilter(Point initialPoint)
{
	int nStates = 5, nMeasurements = 2, nInputs = 1;
	KF = KalmanFilter(nStates, nMeasurements, nInputs, CV_64F);
	state.create(nStates, 1, CV_64F);/* (x, y, Vx, Vy, a) */
    measurementNoise.create(nMeasurements, 1, CV_64F);
    processNoise.create(nStates, 1, CV_64F);
    measurement.create(nMeasurements,1,CV_64F); measurement.setTo(Scalar(0.0));
    noisyMeasurement.create(nMeasurements,1,CV_64F); noisyMeasurement.setTo(Scalar(0.0));
    prevMeasurement.create(nMeasurements,1,CV_64F); prevMeasurement.setTo(Scalar(0.0));    
    int dt=50, T=1000;
     /// A (TRANSITION MATRIX INCLUDING VELOCITY AND ACCELERATION MODEL)
    KF.transitionMatrix.at<double>(0,0) = 1;
    KF.transitionMatrix.at<double>(0,1) = 0;
    KF.transitionMatrix.at<double>(0,2) = (dt/T);
    KF.transitionMatrix.at<double>(0,3) = 0;
    KF.transitionMatrix.at<double>(0,4) = 0.5*(dt/T)*(dt/T);

    KF.transitionMatrix.at<double>(1,0) = 0;
    KF.transitionMatrix.at<double>(1,1) = 1;
    KF.transitionMatrix.at<double>(1,2) = 0;
    KF.transitionMatrix.at<double>(1,3) = (dt/T);
    KF.transitionMatrix.at<double>(1,4) = 0.5*(dt/T)*(dt/T);

    KF.transitionMatrix.at<double>(2,0) = 0;
    KF.transitionMatrix.at<double>(2,1) = 0;
    KF.transitionMatrix.at<double>(2,2) = 1;
    KF.transitionMatrix.at<double>(2,3) = 0;
    KF.transitionMatrix.at<double>(2,4) = (dt/T);

    KF.transitionMatrix.at<double>(3,0) = 0;
    KF.transitionMatrix.at<double>(3,1) = 0;
    KF.transitionMatrix.at<double>(3,2) = 0;
    KF.transitionMatrix.at<double>(3,3) = 1;
    KF.transitionMatrix.at<double>(3,4) = (dt/T);

    KF.transitionMatrix.at<double>(4,0) = 0;
    KF.transitionMatrix.at<double>(4,1) = 0;
    KF.transitionMatrix.at<double>(4,2) = 0;
    KF.transitionMatrix.at<double>(4,3) = 0;
    KF.transitionMatrix.at<double>(4,4) = 1;

    /// Initial estimate of state variables
    KF.statePost = cv::Mat::zeros(nStates, 1,CV_64F);
    KF.statePost.at<double>(0) = initialPoint.x;
    KF.statePost.at<double>(1) = initialPoint.y;
    KF.statePost.at<double>(2) = 0.1;
    KF.statePost.at<double>(3) = 0.1;
    KF.statePost.at<double>(4) = 0.1;

    KF.statePre = KF.statePost;
    state = KF.statePost;

    /// Ex or Q (PROCESS NOISE COVARIANCE)
    setIdentity(KF.processNoiseCov, Scalar::all(0.1));
    /// Initial covariance estimate Sigma_bar(t) or P'(k)
    setIdentity(KF.errorCovPre, Scalar::all(0.1));
    /// Sigma(t) or P(k) (STATE ESTIMATION ERROR COVARIANCE)
    setIdentity(KF.errorCovPost, Scalar::all(0.1));

    /// B (CONTROL MATRIX)
    KF.controlMatrix = cv::Mat(nStates, nInputs,CV_64F);
    KF.controlMatrix.at<double>(0,0) = /*0.5*(dt/T)*(dt/T);//*/0;
    KF.controlMatrix.at<double>(1,0) = /*0.5*(dt/T)*(dt/T);//*/0;
    KF.controlMatrix.at<double>(2,0) = 0;
    KF.controlMatrix.at<double>(3,0) = 0;
    KF.controlMatrix.at<double>(4,0) = 1;

    /// H (MEASUREMENT MATRIX)
    KF.measurementMatrix = cv::Mat::eye(nMeasurements, nStates, CV_64F);

    /// Ez or R (MEASUREMENT NOISE COVARIANCE)
    setIdentity(KF.measurementNoiseCov, Scalar::all(0.1));
}

MyKalmanFilter::~MyKalmanFilter()
{

}
void MyKalmanFilter::Measure(Point measurementPoint)
{
	/// STATE UPDATE
	Mat prediction = KF.predict();

	/// MAKE A MEASUREMENT
	measurement.at<double>(0) = measurementPoint.x;
	measurement.at<double>(1) = measurementPoint.y;

	/// MEASUREMENT NOISE SIMULATION
	randn( measurementNoise, Scalar(0),Scalar::all(sqrtf(0.1)));
	noisyMeasurement = measurement + measurementNoise;

	/// MEASUREMENT UPDATE
	Mat estimated = KF.correct(noisyMeasurement);
	
	noisyPt = Point(noisyMeasurement.at<double>(0),noisyMeasurement.at<double>(1));
	estimatedPt = Point(estimated.at<double>(0),estimated.at<double>(1));
	measuredPt = Point(measurement.at<double>(0),measurement.at<double>(1));
}