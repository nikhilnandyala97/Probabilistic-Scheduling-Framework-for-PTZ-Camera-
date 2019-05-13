#include "motion.h"



//global variable are explained here.
Size video_size;
int mask_add_step = 50;

double BG_scale_factor = 0.5f;
double Body_scale_factor = 0.7f;
double Face_scale_factor = 0.8f;

//Number of frames required for analysis w .

int accumulate_time = 5;

//The Number of frames for which it must zoom.

int display_time; //defualt 150


double time_zoomin_percent = 0.28;
double time_stay_percent = 0.22;
double time_zoomout_percent = 0.3;

// static struct timeval tm;
double time_pause_percent = 0.2;










int main( int argc, const char **argv )
{
    //Timer is being set

    clock_t start, end;
    start = clock();
    srand(time(NULL));
    
    int NoStates= atoi(argv[1]);
    display_time = atoi(argv[2]);
    
   // All the required Matrices are being defined here.
    Mat frame, fore, frame_hsv, frame_hue, mask_inRange, hist, backproj, dst, frame_display;

    Mat roi_hsv, roi_hist;

    //local varibales intialization.
       
    int frame_counter = 1;
    int x = 1, y = accumulate_time, N = 1;

    bool mask_switch, kalmanFilter_switch = true;
    //median filter

    //ROI center is being intialsed
  
    vector<int> *rectangle_center_x = new vector<int>();
    vector<int> *rectangle_center_y = new vector<int>();

    //camshift
    int hsize = 16;
    float hranges[] = {0, 180};

    const float *phranges = hranges;
   
    //All the models or algorithms used are being intialised.

    //BackGround Detection :

    Ptr<BackgroundSubtractor> pMog=createBackgroundSubtractorMOG2();;

    
    //Face Detector dnn use caffe proto and model are being loaded 

   // const string proto = findDataFile("dnn/face_detector/deploy.prototxt", false);
    //const string model = findDataFile("dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel", false);
    // net is the neural network using to detect faces.

    dnn::Net net = readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");

    // Body Detector using Hog using SVM Detecto ,both the methods are intilased but you can select which one to use commenting the other.
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    double video_scale_factor = 1.0f;


    // Face Detector using Hog using SVM Detector ,both the methods are intilased but you can select which one to use commenting the other.
    CascadeClassifier face_cascade;
    if ( ! face_cascade.load( "haarcascade_frontalface_alt.xml" ) )
    {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

    
    video_size.width=1280;
    video_size.height=720;
    
   // mask is the matrix where sensitivity_map will be calculated.
      
    Mat mask = Mat::zeros(Size(video_size.width, video_size.height), CV_8UC1);

  // Intialsing Face1 is Face detected by Cascade, Face 2 is Face detected by DNN, BodyI is Human and BodY is modtion detection Matrix.
    Mat Face1 = Mat::zeros(Size(video_size.width, video_size.height), CV_8UC1);
    Mat BodyI = Mat::zeros(Size(video_size.width, video_size.height), CV_8UC1);
    Mat Face2 = Mat::zeros(Size(video_size.width, video_size.height), CV_8UC1);
    Mat BodyY = Mat::zeros(Size(video_size.width, video_size.height), CV_8UC1);
    

    //Penalty map is intialised but we are not using it it is left because it may useful aftwerwards.   
    Mat penalty = Mat::zeros(Size(video_size.width, video_size.height), CV_32FC1);
    
    // Rectangle that we detected later are intilaised here for example max_rect ROI, rect_track.

    Rect max_rect(0, 0, 1, 1), rect_backup(0, 0, 1, 1), rect_track(0, 0, 1, 1), rect_display(0, 0, 1, 1),
         mask_penalty(0, 0, 0, 0), backProj_penalty(0, 0, 0, 0);

    bool penalty_switch = false;
    bool pause_switch = false;

    Rect videoRect(Point(0, 0), Size(video_size.width, video_size.height));
    Size scalingWindow(video_size.width / 5, video_size.height / 5);
    // Windows that show the mask i,e sensitivity, back proj i.e motion and final output of the framework.

    namedWindow("mask", 0);
    namedWindow("backproj", 0);
    namedWindow("finaloutput", 0);
    

    int imageno=0;

    Mat frame_dummy;

    //intialisation switch
    int intialise=1;

    // State ImportanceScore array.
    int StateValues[NoStates];

    // Intialising StateValues.

    for(int i=0;i<NoStates;i++){
        StateValues[i]=0;

      }

     string ipAddress=argv[3];

     //url1 gives the camera video frame at the time the url is called.
     string url1="http://admin:*1Password@"+ipAddress+"/web/tmpfs/snap.jpg";

     //Running Time of the framework.
     int runningtime=atoi(argv[4]);

     // Intilaising state values 
     int CurrState=1;


     // while loop for howmuch time the code must be running.
     while (((double) (clock() - start)) / CLOCKS_PER_SEC<=runningtime*60)
       {

        //frame dummy gets the image using the url.
          frame_dummy=curlImg(url1.c_str(),10);
        // resizing the 1080p to 720p 

          resize(frame_dummy,frame,  Size(1280, 720), 0, 0, INTER_LANCZOS4);

          if (frame.empty())
           {
            cout << "!!! No captured frame" << endl;
            break;
           }


        frame_display = frame.clone();
        if (video_scale_factor != 1.0f)
            resize(frame, frame, Size(), video_scale_factor, video_scale_factor, INTER_CUBIC);

        cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
        frame_hue.create(frame_hsv.size(), frame_hsv.depth());
        
        int ch[] = {0, 0};
        mixChannels(&frame_hsv, 1, &frame_hue, 1, ch, 1);
        inRange(frame_hsv, Scalar(0, 100, 15), Scalar(180, 255, 255), mask_inRange);
        

         // if frame_counter between x and y means it is in the first 5 frames of the cycle.

	if (IsIntegerBetween(frame_counter, x, y))
        {
          
             
            if(frame_counter==x){
           // After intialising all the states i.e. intialsing == Nostates+1 , we intialise all the state values to average of states.
 
                if(intialise==NoStates+1){
                        int total=0;

                       for(int rr=0;rr<NoStates;rr++){
                            total=total+StateValues[rr];
                           }
                           total=total/NoStates;
                        for(int rr=0;rr<NoStates;rr++){
                            StateValues[rr]=total;
                           }
                        intialise=NoStates+20;
                   }
                   
                if(intialise<=NoStates){
                    
                    //insitalise such all the states are oberverd once in the beggining.
         string url="http://admin:*1Password@"+ipAddress+"/cgi-bin/hi3510/param.cgi?cmd=preset&-act=goto&-number="+to_string(intialise);
                   
                   curlptzcam(url.c_str(),10);
                   
                   cout<<""<<endl;
                   cout<<"Intialise :"<<intialise<<endl;
                   cout<<"sleeping"<<endl;

                      //sleeping because to avoid transition state frames.
                   usleep(7000000);
                   cout<<"sleeping finished"<<endl;
                   
		    intialise=intialise+1;
                   
                    CurrState=intialise;
                    
                   }
                 else{
                        //when intialise state is more than N+1 then intialsing is finished.
                        CurrState=nextStateGenerator(StateValues);
                        

                        //checking for userinput if userinput there .
                        ifstream inFile;int state=0;
                        
                        inFile.open("/home/teja/Desktop/S2/UserInput.txt");
			if (!inFile) {
    				cout << "Unable to open file datafile.txt";
    			}
			else{
			   
			   int x=0;
			   bool test=true;
                           // The web program writes the userinput to a file and we read it from here and update our state values.

			   while (inFile >> x) {
  				state = state + x;
				CurrState=state;
				cout<<"Userinput"<<CurrState<<endl;
				// Giving priority to the state
				int sumOfStateValues=0;


				for(int l=0;l<NoStates;l++){
					sumOfStateValues=sumOfStateValues+StateValues[l];
				}

				// 25% of sum of state values is given to the userinput state.

				StateValues[CurrState-1]=0.25*sumOfStateValues;

                   	        test=false;				
				}
				inFile.close();
                                 // after reading we delete it from the file UserInput.
				if(!test){ofstream myfile;
				 myfile.open ("/home/teja/Desktop/S2/UserInput.txt");
				myfile.close();}
			}
                     

		    for(int ki=0;ki<NoStates;ki++){
                             
			cout<<"State "<<ki+1<<": "<<StateValues[ki]<<";";
                      }
                      
                     // Currstate is p
		    cout<<endl;
                    cout<<"CurrState: "<<CurrState<<endl;

                    //Once the Current State is calculated we visit that state using the following url. 
           string url="http://admin:*1Password@"+ipAddress+"/cgi-bin/hi3510/param.cgi?cmd=preset&-act=goto&-number="+to_string(CurrState);
        
                     curlptzcam(url.c_str(),10);
                      cout<<"transition"<<endl;
                      usleep(7000000);
                      cout<<"transition finished"<<endl;
                   
                   }

                  

		}
            else{
              
        // The mask is added by Motion Detection Multiplied by accuracy , this function updates the mask values of the frame using PMog.  
          
	    BodyY=BackGroundDetection(frame, mask,pMog);
           
       //The mask is added with HumanDetection Muliplied by accuracy ,using hog Human Detection is done.
           BodyI=PeopleDetectByHOG(frame, mask, hog);
        
          //Face1=FaceDetectByCascade(frame, mask, &face_cascade);
           
          //The mask is added with FaceDetection Mat Muliplied by accuracy ,using dnn named net intialised Face Detection is done.
            Face2=FaceDetectorDNN( frame,mask, net);
            
		  }
              
              





            //Displaying the mask i.e the Senstivity MAP
            imshow("mask", mask);
            

              // Final step of the analysing phase in the cycle

            if (frame_counter == y)
            { 

                  // Update the Current Value based on the analysing phase.
                  StateValues[CurrState-1]=0.6*StateValues[CurrState-1]+0.4*SumofPixels(mask);
                

                 x = display_time * N + 1;
                y = x + accumulate_time - 1;
                N++;
                
                 // Finding ROI based on Sensitivity Map i.e. mask.
		max_rect = FindContourAndGetMaxRect2(mask);

                 // Adjusting and calculating the ROI
     
                if (max_rect.width * max_rect.height <= 5000){
                    max_rect = Rect_AdjustSizeAroundCenter(max_rect, 1.3, 1.3, video_size.width, video_size.height);}

		rectangle(frame, max_rect.tl(), max_rect.br(), Scalar(0, 255, 255), 3, 8 , 0);

                rect_track = max_rect;

                Mat roi_max(frame_hue, max_rect), roi_mask(mask, max_rect);

                 Point cor = (rect_track.br() + rect_track.tl())*0.5;
                
                
                
                
                calcHist(&roi_max, 1, 0, roi_mask, roi_hist, 1, &hsize, &phranges);

                normalize(roi_hist, roi_hist, 0, 255, CV_MINMAX);
                
                  // Intialising Mask for the next iteration .
                mask = Mat::zeros(Size(video_size.width, video_size.height), CV_8UC1);

                rectangle_center_x->clear();
                rectangle_center_y->clear();


              
            }
        }
        /* CamShift */
        // frame_counter > accumulate_time means it has come to zooming and tracking on the ROI.
 
        if (frame_counter > accumulate_time)
             {
             //calculating values for zooming
  
       
	     Rect adjusted_rect_track = Rect_AdjustSizeAroundCenter(rect_track, 2, 1.2,
                                       video_size.width, video_size.height);
             
            // calculating backprojection that must be given to MeanShift that tracks our ROI
     
            Mat roi_frame_hue(frame_hue, adjusted_rect_track);
            Mat new_frame_hue = Mat::zeros(frame_hue.size(), frame_hue.depth());
            roi_frame_hue.copyTo(new_frame_hue(adjusted_rect_track));

            calcBackProject(&new_frame_hue, 1, 0, roi_hist, backproj, &phranges, 1, true);

            Mat roi_mask_inRange(mask_inRange, adjusted_rect_track);
            Mat new_mask_inRange = Mat::zeros(mask_inRange.size(), mask_inRange.depth());
            roi_mask_inRange.copyTo(new_mask_inRange(adjusted_rect_track));
            //remove the noise
            backproj &= new_mask_inRange;
            
	    imshow("backproj", backproj);
           
             // Input of backproj to mean shift algorithm
           
            meanShift(backproj, rect_track, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
           
            //the ROI should be 50% larger
            rect_track = GetMedianRectResult(rectangle_center_x, rectangle_center_y, rect_track, 5);
            

	    /*********** tracking part start***************************/

             
            int currentFrame = ((frame_counter - (accumulate_time + 1)) % (display_time)); //[0->149]
            
             // based on back projection meanshift gives its movement and outputs rect_track
    

            // this below function Rect_AdjustSizeAroundCenter takes the rect_track shifts the roi which is given as output to rect_display.
            rect_display = Rect_AdjustSizeAroundCenter(rect_track, 1.3, 1.3, video_size.width, video_size.height);
           
           if (rect_display.width * rect_display.height <= 384 * 216)
            {
                rect_display = Rect_WithCenter(Rect_GetCenter(rect_display), Size(384, 216));
                

            }
 
            rect_display = ChangeRatioOfRect(rect_display, (float)videoRect.width / (float) videoRect.height);
            rect_display = SupplementRect(rect_display, videoRect.width, videoRect.height);
            int tracking_last_frame = display_time * (time_stay_percent + time_zoomin_percent) - 1;

            /******************tracking part ends **********************/


            // Zooming part For zoom in we use Cubic Spline
            
            if (currentFrame <= display_time * time_zoomin_percent) //==> the parameter here need to be fixed
            {
                Rect zoomRect = Zoom_CubicSpline(videoRect,
                                                 rect_display,
                                                 0, //start frame
                                                 display_time * time_zoomin_percent, //end frame
                                                 currentFrame);// current frame
                frame_display = frame_display(zoomRect);
                


            }
            else if (currentFrame <= display_time * (time_stay_percent + time_zoomin_percent))
            {  // the time period betwwen Zoomin and Zoom out.

                frame_display = frame_display(rect_display);
                
                if (currentFrame == tracking_last_frame)
                {
                
		    mask_penalty = Rect_AdjustSizeAroundCenter(rect_track, 2.0f, 1.3f, video_size.width, video_size.height);
                
		    penalty_switch = true;
                }
            }
            else if (currentFrame <= display_time * (time_stay_percent + time_zoomin_percent + time_zoomout_percent))
            {

                // Zooming out using Cubic Spline. 
                Rect zoomRect = Zoom_CubicSpline(rect_display,
                                                 videoRect,
                                                 display_time * (time_stay_percent + time_zoomin_percent),
                                                 display_time * (time_stay_percent + time_zoomin_percent + time_zoomout_percent),
                                                 currentFrame);
                frame_display = frame_display(zoomRect);
            }

            resize(frame_display, frame_display, Size(video_size.width / 5, video_size.height / 5), 0, 0, INTER_CUBIC);
            
             // The FInal output is shown in the window "finaloutput"
            imshow("finaloutput", frame_display);
		imageno=imageno+1;
            //frame output is even store in our local system from where the web reads the images and streams them.

            stringstream ss;
            ss << imageno;
            
	 // String outputPath_str="/home/teja/Desktop/S2/images/"+ss.str()+".png";
          Mat frame_display1;
	  resize(frame_display,frame_display1,  Size(512, 216), 0, 0, INTER_LANCZOS4);
         // imwrite(outputPath_str, frame_display1);
            /*********** tracking part end ****************/
        }
        else
        {
            resize(frame_display, frame_display, Size(video_size.width / 5, video_size.height / 5), 0, 0, INTER_CUBIC);
            

        }
        
	frame_counter++;
        
         // waiting 10 milliseconds between each frame for transition to be natural.

	char c = (char)waitKey(10);
        if ( c == 27 ){ 
              cout<<"Break";
             break;
          }
    
      }


    // waitKey(0);
    destroyAllWindows();
    return 0;
}

