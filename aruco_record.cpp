/**
Copyright 2017 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
*/

#include "CameraApi.h"
#include "aruco.h"
#include "timers.h"
#include "date.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/Geometry>
#include "sglviewer.h"
using namespace cv;
using namespace aruco;
using namespace std;
string TheMarkerMapConfigFile;
bool The3DInfoAvailable = false;
float TheMarkerSize = -1;
VideoCapture TheVideoCapturer;
Mat TheInputImage, TheInputImageCopy;
CameraParameters TheCameraParameters;
MarkerMap TheMarkerMapConfig;
MarkerDetector TheMarkerDetector;
MarkerMapPoseTracker TheMSPoseTracker;

unsigned char           * g_pRgbBuffer;
int                     iCameraCounts = 1;
int                     iStatus=-1;
tSdkCameraDevInfo       tCameraEnumList;
int                     hCamera;
tSdkCameraCapbility     tCapability;      //设备描述信息
tSdkFrameHead           sFrameInfo;
BYTE*			        pbyBuffer;
int                     iDisplayFrames = 10000;
IplImage *iplImage = NULL;
int                     channel=3;

int waitTime = 0;
std::map<int, cv::Mat> frame_pose_map;  // set of poses and the frames they were detected
sgl_OpenCV_Viewer Viewer;


int main(int argc, char** argv)
{
    try
    {
        CameraSdkInit(1);

        iStatus = CameraEnumerateDevice(&tCameraEnumList,&iCameraCounts);
        iStatus = CameraInit(&tCameraEnumList,-1,-1,&hCamera);
        g_pRgbBuffer = (unsigned char*)malloc(3088*2064*3);
        CameraPlay(hCamera);
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_BGR8);
        CameraSetGain(hCamera, 100, 100, 100);

        waitTime = 10;
        
        TheVideoCapturer.set(CV_CAP_PROP_FRAME_WIDTH, 3088);
        TheVideoCapturer.set(CV_CAP_PROP_FRAME_HEIGHT, 2064);

        VideoWriter video("video.mp4", CV_FOURCC('F','M','P','4'), 10, Size(3088,2064)); 
        namedWindow("Recording...", WINDOW_NORMAL);
        resizeWindow("Recording...", 800, 535);

        while(true)
        {
            CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000);
            CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer,&sFrameInfo);
            
            cv::Mat TheInputImage(
                cvSize(sFrameInfo.iWidth,sFrameInfo.iHeight), 
                sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
                g_pRgbBuffer
            );

            CameraReleaseImageBuffer(hCamera,pbyBuffer);
            
            video.write(TheInputImage);

            imshow("Recording...", TheInputImage);

            // Press  ESC on keyboard to  exit
            char c = (char)waitKey(1);
            if( c == 27 ) 
                break;
        }

        // When everything done, release the video capture and write object
        video.release();
        
        // Closes all the windows
        destroyAllWindows();
        return 0;
    }
    catch (std::exception& ex)

    {
        cout << "Exception :" << ex.what() << endl;
    }
}