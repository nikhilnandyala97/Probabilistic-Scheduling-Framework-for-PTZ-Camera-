#include "motion.h"
int CalTimeDifference(struct timeval tm1, struct timeval tm2)
{
    return 1000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec) / 1000;
}



string GetCurrentDateTime()
{
    time_t now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    // strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}



bool IsIntegerBetween(int number, int low, int high)
{
    return (number >= low && number <= high);
}
/**
 * With find convex hull
 * @param
 * @return
 */



Rect FindContourAndGetMaxRect2(Mat frame)
{
    //add find contours
    Rect maxRect(0, 0, 1, 1);
    Mat temp = frame.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    int most_important_area = 0;
    int max = 0;
    Mat drawing = Mat::zeros( temp.size(), CV_8UC1 );
    for ( int i = 0; i < contours.size(); i++ )
    {
        drawContours( drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy);
        int temp_sum_pixels = SumAllPixelsInRect(frame, boundingRect(contours[i]));
        if (most_important_area < temp_sum_pixels)
        {
            most_important_area = temp_sum_pixels;
            max = i;
        }
    }
    if (contours.size() != 0)
    {
        maxRect  = boundingRect(contours[max]);
        rectangle(drawing, maxRect.tl(), maxRect.br(), Scalar(255, 255, 255), 2);
       
    }


    return maxRect;


}










Rect Rect_WithCenter(Point center, Size orginalRect)
{
    return Rect(Point(center.x - orginalRect.width / 2,
                      center.y - orginalRect.height / 2),
                orginalRect);
}



template <typename T> string NumberToString( T Number )
{
    ostringstream oss;
    oss << Number;
    return oss.str();
}





size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    vector<uchar> *stream = (vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}




//function to retrieve the image as cv::Mat data type
Mat curlImg(const char *img_url, int timeout)
{
    timeout=10;
    vector<uchar> stream;
    CURL *curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs, 
    CURLcode res = curl_easy_perform(curl); // start curl
    curl_easy_cleanup(curl); // cleanup
    return imdecode(stream, -1); // 'keep-as-is'
}



 void curlptzcam(const char *img_url,int to)
{
    to=10;

    CURL *curl1 = curl_easy_init();
    curl_easy_setopt(curl1, CURLOPT_URL, img_url); //the
         curl_easy_perform(curl1);
         curl_easy_cleanup(curl1);
}






























/* Back ground detection*/ 
vector <Rect> BackGroundDetectionInit(Mat frame, Mat mask,Ptr<BackgroundSubtractor> pMog )
{
    // cout << "enter Background" << endl;
    //copy and resize
    Mat frame_copy = frame.clone(), fore;
    if (BG_scale_factor != 1.0f)
        resize(frame_copy, frame_copy, Size(), BG_scale_factor, BG_scale_factor, INTER_AREA);
    pMog->apply(frame_copy, fore);
    vector<vector<Point> > contours;
    erode(fore, fore, Mat());
    dilate(fore, fore, Mat());
    findContours(fore, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE) ;
    vector<Rect> boundRect( contours.size() );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Vec4i> hierarchy;
    Mat drawing = Mat::zeros(fore.size(), CV_8UC1);
    // double smallest_area  = contourArea( contours[0],false);
    Mat drawing2 = Mat::zeros(fore.size(), CV_8UC1);
    for ( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );

        if ( (contourArea( contours[i], false) >= 100)
                && (contourArea( contours[i], false) < video_size.area() * 0.90))
        {
            Rect r = Rect_AdjustSizeAroundCenter(boundRect[i], 0.45, 0.6);
            boundRect[i]=r;
           rectangle( drawing2, r.tl(), r.br(), Scalar(255, 255, 255), -1, 8, 0);
            drawContours( drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy);
        }
        //the min_size and max_size here should be fixed
        // if(boundRect[i].area() >= 100 && boundRect[i].area() < video_size.area() * 0.95)
        // {
        //   // rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(255,255,255), -1, 8, 0);
        //       drawContours( drawing, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy);
        // }
    }
    // imshow("drawing-scale",drawing);
    if (BG_scale_factor != 1.0f)
        resize(drawing, drawing, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);
    // imshow("drawing-original",drawing);

      if (BG_scale_factor != 1.0f)
        resize(drawing2, drawing2, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);
    // imshow("drawing-original",drawing);


    for (int i = 0; i < mask.cols; i++)
    {
        for (int j = 0; j < mask.rows; j++)
        {
            Point p = Point(i, j);
            if (drawing.at<uchar>(p) == 255)
                mask.at<uchar>(p) += mask_add_step;
        }
    }
return boundRect;

}







/* Back ground detection*/ 
Mat BackGroundDetection(Mat frame, Mat mask,Ptr<BackgroundSubtractor> pMog )
{
    // cout << "enter Background" << endl;
    //copy and resize
    Mat frame_copy = frame.clone(), fore;
    if (BG_scale_factor != 1.0f)
        resize(frame_copy, frame_copy, Size(), BG_scale_factor, BG_scale_factor, INTER_AREA);
    pMog->apply(frame_copy, fore);
    vector<vector<Point> > contours;
    erode(fore, fore, Mat());
    dilate(fore, fore, Mat());
    findContours(fore, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE) ;
    vector<Rect> boundRect( contours.size() );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Vec4i> hierarchy;
    Mat drawing = Mat::zeros(fore.size(), CV_8UC1);
    // double smallest_area  = contourArea( contours[0],false);
    Mat drawing2 = Mat::zeros(fore.size(), CV_8UC1);
    for ( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );

        if ( (contourArea( contours[i], false) >= 100)
                && (contourArea( contours[i], false) < video_size.area() * 0.90))
        {
        	Rect r = Rect_AdjustSizeAroundCenter(boundRect[i], 0.45, 0.6);
        	boundRect[i]=r;
           rectangle( drawing2, r.tl(), r.br(), Scalar(255, 255, 255), -1, 8, 0);
            drawContours( drawing, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy);
        }
      

    }

    if (BG_scale_factor != 1.0f)
        resize(drawing, drawing, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);


      if (BG_scale_factor != 1.0f)
        resize(drawing2, drawing2, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);



    for (int i = 0; i < mask.cols; i++)
    {
        for (int j = 0; j < mask.rows; j++)
        {
            Point p = Point(i, j);
            if (drawing.at<uchar>(p) == 255)
                mask.at<uchar>(p) += mask_add_step;
        }
    }
return drawing2;

}





/*

Mat HumanDetectionYOLO( Mat frame , Mat mask, Net onet){


Mat frame_copy = frame.clone();

resize(frame_copy, frame_copy,Size(oWidth, oHeight), 0, 0, INTER_AREA);

Mat blob=blobFromImage(frame_copy, 1/255.0, cvSize(oWidth, oHeight), Scalar(0,0,0), true, false);

onet.setInput(blob);

vector <Mat> outs;

onet.forward(outs,getOutputsNames(onet));




vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
     
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold && classIdPoint.x==0)
            {
                int centerX = (int)(data[0] * frame_copy.cols);
                int centerY = (int)(data[1] * frame_copy.rows);
                int width = (int)(data[2] * frame_copy.cols);
                int height = (int)(data[3] * frame_copy.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                 
                //classIds.push_OuputRunningTimeback(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

Mat drawing = Mat::zeros(frame_copy.size(), CV_8UC1);
    vector<Rect>  found_filtered;


 size_t i, j;
    for ( i = 0; i < boxes.size(); i++ )
    {
        Rect r = boxes[i];
        for ( j = 0; j < boxes.size(); j++ )
            if ( j != i && (r & boxes[j]) == r)
                break;
        if ( j == boxes.size() )
            found_filtered.push_back(r);
    }

   


for ( i = 0; i < found_filtered.size(); i++ )
    {
        Rect r = found_filtered[i];
        r = Rect_AdjustSizeAroundCenter(r, 0.45, 0.6);
        rectangle( drawing, r.tl(), r.br(), Scalar(255, 255, 255), -1, 8, 0);
       

    }

 resize(drawing, drawing, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);

    return drawing;
}

*/






long double SumofPixels(Mat mask ){

long double total=0;
long double dummy=0;
for (int i = 0; i < mask.cols; i++)
    {
        for (int j = 0; j < mask.rows; j++)
        {
            Point p = Point(i, j);
                dummy=( mask.at<uchar>(p)/255.001);
            
                total=total+dummy ;
               
        }
    }



return total;


}

int nextStateGenerator(int A[]){
	srand(time(0));
	float sum=0.0;
	for(int i=0;i<=7;i++){
		sum=sum+A[i];
	}
	double random = (double)rand() / (double)RAND_MAX;
	float start=0;
	float end=A[0]/(sum);
	int nextState=0;
	for(int i=0;i<=6;i++){
		if(start<=random && end>=random){
			nextState=i+1;
			cout<<nextState<<endl;
			break;
		}
		start=end;
		end=end+A[i+1]/sum;
	}
	if(start<=random && end>=random&&nextState==0){
	
		nextState=8;
	}
	if(nextState==0)
		nextState=rand()%9+1;
		return nextState;
}





vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}














Mat PeopleDetectByHOG( Mat frame , Mat mask, HOGDescriptor hog)
{
    Mat frame_copy = frame.clone();
   // Mat gray;
    //cvtColor(frame, gray, CV_BGR2GRAY);
    
    if (Body_scale_factor != 1.0f)
    {
        resize(frame_copy, frame_copy, Size(), Body_scale_factor, Body_scale_factor, INTER_AREA);
        // resize(mask, mask, Size(), Body_scale_factor, Body_scale_factor, INTER_AREA);
    }
    // imshow("test1",frameCopy);
    vector<Rect> found, found_filtered;
    //do detection
    

    //if the size of src image is too small, the program will be error as assertion fault
    //the parameters here should be adjusted
    hog.detectMultiScale(frame_copy, found, 0, Size(8, 8), Size(32, 32), 1.059, 0);
    // hog.detectMultiScale(frameCopy, found);
    //remove nested rectangle
    size_t i, j;
    for ( i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];
        for ( j = 0; j < found.size(); j++ )
            if ( j != i && (r & found[j]) == r)
                break;
        if ( j == found.size() )
            found_filtered.push_back(r);
    }

    Mat drawing = Mat::zeros(frame_copy.size(), CV_8UC1);
    for ( i = 0; i < found_filtered.size(); i++ )
    {
        Rect r = found_filtered[i];
        // the HOG detector returns slightly larger rectangles than the real objects.
        // so we slightly shrink the rectangles to get a nicer output.
        r = Rect_AdjustSizeAroundCenter(r, 0.45, 0.6);
        rectangle( drawing, r.tl(), r.br(), Scalar(255, 255, 255), -1, 8, 0);
        // rectangle( mask, r.tl(), r.br(), Scalar(255,255,255), -1, 8, 0);
    }
    if (Body_scale_factor != 1.0f)
    {
        resize(drawing, drawing, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);
        // resize(mask, mask, Size(), 1/Body_scale_factor, 1/Body_scale_factor, INTER_NEAREST);
    }
    
    for (int i = 0; i < mask.cols; i++)
    {
        for (int j = 0; j < mask.rows; j++)
        {
            Point p = Point(i, j);
            if (drawing.at<uchar>(p) == 255)
                mask.at<uchar>(p) += mask_add_step;
               
        }
    }
    return drawing;
}




Mat FaceDetectorDNN(Mat frame, Mat mask ,dnn::Net net){

Mat frame_copy=frame.clone();
// dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
if (frame_copy.channels() == 4){
            cvtColor(frame_copy, frame_copy, COLOR_BGRA2BGR);
        }
Mat frame_gray;

resize(frame_copy, frame_gray,Size(inWidth, inHeight), 0, 0, INTER_AREA);
    

Mat inputBlob = blobFromImage(frame_gray, inScaleFactor,Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images

 net.setInput(inputBlob, "data");


 Mat detection = net.forward("detection_out"); //compute output

 float confidenceThreshold = 0.4;

  Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
  vector<Rect> detectedRects;
for (int i = 0; i < detectionMat.rows; i++)
{
    float confidence = detectionMat.at<float>(i, 2);
    if (confidence > confidenceThreshold )
    {
        int objectClass = (int)(detectionMat.at<float>(i, 1));
        int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame_gray.cols);
        int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame_gray.rows);
        int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame_gray.cols);
        int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame_gray.rows);

        Rect box(Point(left, top), Point(right, bottom));
       detectedRects.push_back(box);
    }

}
 cout<<detectedRects.size()<<endl;

    Mat drawing = Mat::zeros(frame_gray.size(), CV_8UC1);
    for ( size_t i = 0; i < detectedRects.size(); i++ )
    {
        //draw the face
        //Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        //ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        //slightly  adjust the size of the face
        detectedRects[i] = Rect_AdjustSizeAroundCenter(detectedRects[i], 0.8, 0.8);
        rectangle( drawing, detectedRects[i].tl(), detectedRects[i].br(), Scalar(255, 255, 255), -1, 8, 0);
    }

    resize(drawing, drawing, Size(video_size.width, video_size.height), 0, 0, INTER_NEAREST);
    for (int i = 0; i < drawing.cols; i ++)
    {
        for (int j = 0; j < drawing.rows; j++)
        {
            Point p = Point(i, j);
            if (drawing.at<uchar>(p) == 255)
                mask.at<uchar>(p) += mask_add_step;
        }
    }






return drawing;

}












Mat FaceDetectByCascade(Mat frame, Mat mask, CascadeClassifier *face_cascade)
{
    Mat frame_copy = frame.clone();
    if (Face_scale_factor != 1)
        resize(frame_copy, frame_copy, Size(), Face_scale_factor, Face_scale_factor, INTER_AREA);


    Mat frame_gray;
    cvtColor( frame_copy, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    vector<Rect> faces;
    face_cascade->detectMultiScale( frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(28, 28) );
    Mat drawing = Mat::zeros(frame_gray.size(), CV_8UC1);
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        
	faces[i] = Rect_AdjustSizeAroundCenter(faces[i], 0.8, 0.8);
        rectangle( drawing, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), -1, 8, 0);
    }

    if (Face_scale_factor != 1)
        resize(drawing, drawing, Size(), 1 / Face_scale_factor, 1 / Face_scale_factor, INTER_NEAREST);
   // FaceI=drawing;
  
    for (int i = 0; i < drawing.cols; i ++)
    {
        for (int j = 0; j < drawing.rows; j++)
        {
            Point p = Point(i, j);
            if (drawing.at<uchar>(p) == 255)
                mask.at<uchar>(p) += mask_add_step;
        }
    }

return drawing;
}







Rect Rect_AdjustSizeAroundCenter(Rect rect, double width, double height,
                                 int videoWidth, int videoHeight)
{
    if (videoWidth > 0 && videoHeight > 0)
    {
        if (width < 1 && height < 1)
        {
            return Rect(rect.x + (rect.width - rect.width * width) / 2,
                        rect.y + (rect.height - rect.height * height) / 2,
                        rect.width * width,
                        rect.height * height);
        }
        else if (width > 1 && height > 1)
        {
            Rect tempRect = Rect(rect.x - (rect.width * width - rect.width) / 2,
                                 rect.y - (rect.height * height - rect.height) / 2,
                                 rect.width * width,
                                 rect.height * height);
            int x1 = (tempRect.tl().x < 0) ? 0 : tempRect.tl().x;
            int y1 = (tempRect.tl().y < 0) ? 0 : tempRect.tl().y;
            int x2 = (tempRect.br().x > videoWidth - 1) ? videoWidth : tempRect.br().x;
            int y2 = (tempRect.br().y > videoHeight - 1) ? videoHeight : tempRect.br().y;
          
	  return Rect(Point(x1, y1), Point(x2, y2));
        }
        else if (width < 1 && height > 1)
        {
            //do something
        }
        else if (width > 1 && height < 1)
        {
            //do something
        }
    }
    else if (videoWidth == 0 && videoHeight == 0)
    {
        if (width < 1 && height < 1)
        {
            return Rect(rect.x + (rect.width - rect.width * width) / 2,
                        rect.y + (rect.height - rect.height * height) / 2,
                        rect.width * width,
                        rect.height * height);
        }
        else if (width > 1 && height > 1)
        {
            int x = rect.x - (rect.width * width - rect.width) / 2;
            int y = rect.y - (rect.height * height - rect.height) / 2;
            int rect_width =  rect.width * width;
            int rect_height = rect.height * height;
            return Rect(x, y, rect_width, rect_height);
        }
        else if (width < 1 && height > 1)
        {
            //do something
        }
        else if (width > 1 && height < 1)
        {
            //do something
        }
    }
}


//MediumRect and smallRect are both in the outsidRect
Rect Rect_AdjustRelativePosition(Rect mediumRect, Rect smallRect)
{
    return Rect(smallRect.x - mediumRect.x,
                smallRect.y - mediumRect.y,
                smallRect.width,
                smallRect.height);
}
//mediumRect is in the outside Rect, the point is relativePoint in the mediumRect
Point Point_AdjustRelativePosition(Rect mediumRect, Point point)
{
    return Point(point.x + mediumRect.x, point.y + mediumRect.y);
}



Point Rect_GetCenter(Rect rectangle)
{
    return Point(rectangle.x + rectangle.width / 2,
                 rectangle.y + rectangle.height / 2);
}





void merge(int *s, int m, int k, int n)
{
    int sizeA = k - m + 1;
    int sizeB = n - k;
    int size = sizeA + sizeB;
    int *c = (int *)malloc(sizeof(int) * size);

    int ia = m;
    int ib = k + 1;
    int tmpA;
    int tmpB;
    for (int i = 0; i < size; i++)
    {
        if (ia > k) tmpA = INT_MAX;
        else tmpA = s[ia];
        if (ib > n) tmpB = INT_MAX;
        else tmpB = s[ib];

        if (tmpA <= tmpB)
        {
            c[i] = s[ia];
            ia++;
        }
        else
        {
            c[i] = s[ib];
            ib++;
        }
    }

    memcpy(s + m, c, sizeof(int)*size);
    free(c);
    return;
}



void mergeSort(int *s, int m, int n)
{
    if (m == n) return;
    int mid = (n - m) / 2 + m;
    mergeSort(s, m, mid);
    mergeSort(s, mid + 1, n);
    merge(s, m, mid, n);
    return;
}


//list or forward list is more efficient than using vector
//set window size


int GetMedianResult(vector<int> *v, int insert, int window)
{
    v->push_back(insert);
    // for(int i = 0; i < v->size(); i++)
    //   cout << (*v)[i] << " ";
    // cout << endl;
    vector<int> v_sort = *v;
    sort(v_sort.begin(), v_sort.end());
    int result = v_sort[(v_sort.size() - 1) / 2]  ;
    if (v->size() == window)
    {
        v->erase(v->begin());
    }
    return result;
}



Rect GetMedianRectResult(vector<int> *x, vector<int> *y, Rect rect , int window)
{
    Point center = Rect_GetCenter(rect);
    center.x = GetMedianResult(x, center.x, window);
    center.y = GetMedianResult(y, center.y, window);
    return Rect_WithCenter(center, rect.size());
}


int CalculateInterpolation(float interpolant, int a, int b)
{
    // return interpolant * a + (1 - interpolant) * b;
    return (2 * interpolant * interpolant * interpolant  - 3 * interpolant * interpolant + 1) * b
           + (-2 * interpolant * interpolant * interpolant + 3 * interpolant * interpolant) * a;
}



Rect Zoom_CubicSpline(Rect initialRect, Rect lastRect,
                      int startFrame, int endFrame, int currentFrame)
{
    float interpolant = (float)(currentFrame - startFrame) / (float)(endFrame - startFrame);
    int x = CalculateInterpolation(interpolant, lastRect.tl().x, initialRect.tl().x);
    int y = CalculateInterpolation(interpolant, lastRect.tl().y, initialRect.tl().y);
    int width = CalculateInterpolation(interpolant, lastRect.width, initialRect.width);
    int height = CalculateInterpolation(interpolant, lastRect.height, initialRect.height);
    return Rect(x, y, width, height);
}



Rect ChangeRatioOfRect(Rect zoomRect, float targetRectRatio)
{
  float zoomRectRatio = (float)zoomRect.width / (float)zoomRect.height;
  if(targetRectRatio > zoomRectRatio)
  {
    //increase zoomRect.width 
    int newWidth = targetRectRatio * zoomRect.height;
    int supplementaryWidth = (newWidth - zoomRect.width)/2;
    return Rect(zoomRect.x - supplementaryWidth,
                zoomRect.y,
                newWidth,
                zoomRect.height);

  }else{
    int newHeight = zoomRect.width / targetRectRatio;
    int supplementaryHeight = (newHeight - zoomRect.height)/ 2;
    return Rect(zoomRect.x,
                zoomRect.y - supplementaryHeight,
                zoomRect.width,
                newHeight);
  }
  

}



Rect ChangeRatioOfRectCut(Rect zoomRect, Rect targetRect)
{
    float zoomRectRatio = (float)zoomRect.width / (float)zoomRect.height;
    float targetRectRatio = (float)targetRect.width / (float)targetRect.height;
    if (targetRectRatio > zoomRectRatio)
    {
        
	int newWidth = targetRectRatio * zoomRect.height;
        int supplementaryWidth = (newWidth - zoomRect.width) / 2;
        Rect rect(zoomRect.x - supplementaryWidth, zoomRect.y, newWidth, zoomRect.height);
        int x1 = (rect.tl().x < 0) ? 0 : rect.tl().x;
        int y1 = (rect.tl().y < 0) ? 0 : rect.tl().y;
        int x2 = (rect.br().x > video_size.width - 1) ? video_size.width : rect.br().x;
        int y2 = (rect.br().y > video_size.height - 1) ? video_size.height : rect.br().y;
        return Rect(Point(x1, y1), Point(x2, y2));
       

    }
    else
    {
       
	 int newHeight = zoomRect.width / targetRectRatio;
        int supplementaryHeight = (newHeight - zoomRect.height) / 2;
        int newY = zoomRect.y - supplementaryHeight;
        Rect rect(zoomRect.x, newY, zoomRect.width, newHeight);
        int x1 = (rect.tl().x < 0) ? 0 : rect.tl().x;
        int y1 = (rect.tl().y < 0) ? 0 : rect.tl().y;
        int x2 = (rect.br().x > video_size.width - 1) ? video_size.width : rect.br().x;
        int y2 = (rect.br().y > video_size.height - 1) ? video_size.height : rect.br().y;
        return Rect(Point(x1, y1), Point(x2, y2));
    }
}





double TwoDimensionalGaussianfunction(double A, double centerX, double centerY,
                                      double eX, double eY, double currentX, double currentY)
{
    return A * exp(-(
                       (currentX - centerX) * (currentX - centerX) / (2 * eX * eX)
                       +
                       (currentY - centerY) * (currentY - centerY) / (2 * eY * eY)));
}





void CalculateGasussianROI(Mat_<float> penalty, Rect maxRect, double amplitude)
{
    int halfWidth = maxRect.width / 2;
    int halfHeight = maxRect.height / 2;
    Point maxRectCenter = Rect_GetCenter(maxRect);

    for (int i = maxRect.x; i < (maxRect.x + maxRect.width); i++)
    {
        for (int j = maxRect.y; j < (maxRect.y + maxRect.height); j++)
        {
            penalty.at<float>(Point(i, j)) = TwoDimensionalGaussianfunction(amplitude,
                                             maxRectCenter.x,
                                             maxRectCenter.y,
                                             halfWidth,
                                             halfHeight,
                                             i,
                                             j);
        }
    }

}



void MultiplyFloatInRect(Mat_<float> frame, float value, Rect rect)
{
    int width = rect.x + rect.width;
    int height = rect.y + rect.height;
    for (int i = rect.x; i < width; i++)
    {
        for (int j = rect.y; j < height; j++)
        {
            frame.at<float>(Point(i, j)) = frame.at<float>(Point(i, j)) * value;
        }
    }
}



void SubstractByFloatInRect(Mat_<float> frame, float value, Rect rect)
{
    for (int i = rect.x; i < rect.x + rect.width; i++)
    {
        for (int j = rect.y; j < rect.y + rect.height; j++)
        {
            frame.at<float>(Point(i, j)) =  value - frame.at<float>(Point(i, j));
        }
    }
}




int SumAllPixelsInRect(Mat frame, Rect rect)
{
    int sum = 0, x = rect.x, y = rect.y;
    int width = x + rect.width;
    int height = y + rect.height;
    //cout << frame << endl;
    for (int i = x; i < width; i++)
    {
        for (int j = y; j < height; j++)
        {
            sum += frame.at<uchar>(Point(i, j));
        }
    }
    return sum;
}




bool IsBetweenTwoInt(int value,int left, int right)
{
  if(value >= left && value <= right){
    return true;
  }else
    return false;
}


Rect SupplementRect(Rect rect, int videoWidth, int videoHeight)
{

    int topLeftX = rect.x;
    int topLeftY = rect.y;
    int bottomRightX = rect.br().x;
    int bottomRightY = rect.br().y;
    int widthValue = rect.width;
    int heightValue = rect.height;
    Size scalingWindow(rect.width,rect.height);
   
    if(topLeftX < 0 
      && topLeftY < 0 
      && IsBetweenTwoInt(bottomRightX,1,videoWidth)
      && IsBetweenTwoInt(bottomRightY,1,videoHeight)
      ) //in 1
    {
      //cout << "get in 1" << endl; 
      return Rect(Point(0,0),scalingWindow);
    }
    else if(IsBetweenTwoInt(topLeftX,0,videoWidth-1) 
      && topLeftY < 0 
      && IsBetweenTwoInt(bottomRightX,1,videoWidth)
      && IsBetweenTwoInt(bottomRightY,1,videoHeight)) // in 2
    {
      //cout << "get in 2" << endl;  
      return Rect(Point(topLeftX,0),scalingWindow);
    }
    else if(IsBetweenTwoInt(topLeftX,1,videoWidth-1)
      && topLeftY < 0
      && bottomRightX > videoWidth
      && IsBetweenTwoInt(bottomRightY,1,videoHeight)) // in 3
    {
      //cout << "get in 3" << endl;  
      return Rect(Point(videoWidth - widthValue,0), scalingWindow);
    }
    else if(topLeftX < 0 
      && IsBetweenTwoInt(topLeftY,0,videoHeight-1)
      && IsBetweenTwoInt(bottomRightX,1,videoWidth)
      && IsBetweenTwoInt(bottomRightY,1,videoHeight)) // in 4
    {
     // cout << "get in 4" << endl; 
      return Rect(Point(0,topLeftY),scalingWindow); 
    }
    else if(IsBetweenTwoInt(topLeftX,0,videoWidth-1)
      && IsBetweenTwoInt(topLeftY,0,videoHeight-1)
      && IsBetweenTwoInt(bottomRightX,1,videoWidth)
      && IsBetweenTwoInt(bottomRightY,1,videoHeight)) // in 5
    {
      //cout << "get in 5" << endl; 
      return Rect(Point(topLeftX,topLeftY),scalingWindow);
    }
    else if(IsBetweenTwoInt(topLeftX,0,videoWidth-1)
      && IsBetweenTwoInt(topLeftY,0,videoHeight-1)
      && bottomRightX > videoWidth
      && IsBetweenTwoInt(bottomRightY,1,videoHeight)) // in 6
    {
      //cout << "get in 6" << endl; 
      return Rect(Point(videoWidth - widthValue,topLeftY),scalingWindow);
    }
    else if(topLeftX < 0 
      && IsBetweenTwoInt(topLeftY,1,videoHeight-1)
      && IsBetweenTwoInt(bottomRightX,1,videoWidth)-1
      && bottomRightY > videoHeight) // in 7
    {
     // cout << "get in 7" << endl; 
      return Rect(Point(0,videoHeight-heightValue),scalingWindow);
    }
    else if(IsBetweenTwoInt(topLeftX,0,videoWidth)
      && IsBetweenTwoInt(topLeftY,1,videoHeight-1)
      && IsBetweenTwoInt(bottomRightX,1,videoWidth)
      && bottomRightY > 0) // in 8
    {
      //cout << "get in 8" << endl; 
      return Rect(Point(topLeftX,videoHeight-heightValue),scalingWindow);
    }
    else if(IsBetweenTwoInt(topLeftX,0,videoHeight-1)
      && IsBetweenTwoInt(topLeftY,1,videoHeight-1)
      && bottomRightX > videoWidth
      && bottomRightY > videoHeight) // in 9
    {
      //cout << "get in 9" << endl; 
      return Rect(Point(videoWidth - widthValue,videoHeight - heightValue),scalingWindow);
    }
}
