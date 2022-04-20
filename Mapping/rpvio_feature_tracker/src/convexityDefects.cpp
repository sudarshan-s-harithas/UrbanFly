#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat img, frame, img2, img3;
    VideoCapture cam(0);
    while (true){
        cam.read(frame);

        cvtColor(frame, img, CV_BGR2HSV);

        //thresholding 
        inRange(img, Scalar(0, 143, 86), Scalar(39, 255, 241), img2);

        imshow("hi", img2);

        //finding contours
        vector<vector<Point>> Contours;
        vector<Vec4i> hier;
        //morphological transformations
        erode(img2, img2, getStructuringElement(MORPH_RECT, Size(3, 3)));
        erode(img2, img2, getStructuringElement(MORPH_RECT, Size(3, 3)));

        dilate(img2, img2, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(img2, img2, getStructuringElement(MORPH_RECT, Size(8, 8)));

        //finding the contours required
        findContours(img2, Contours, hier, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0));

        //finding the contour of largest area and storing its index
        int lrgctridx = 0;
        int maxarea = 0;
        for (int i = 0; i < Contours.size(); i++)
        {
            double a = contourArea(Contours[i]);
            if (a> maxarea)
            {
                maxarea = a;
                lrgctridx = i;
            }
        }
        //convex hulls
        vector<vector<Point> >hull(Contours.size());
        vector<vector<int> > hullsI(Contours.size()); 
        vector<vector<Vec4i>> defects(Contours.size());
        for (int i = 0; i < Contours.size(); i++)
        {
            convexHull(Contours[i], hull[i], false);
            convexHull(Contours[i], hullsI[i], false); 
            if(hullsI[i].size() > 3 )            
            {
                convexityDefects(Contours[i], hullsI[i], defects[i]);
            }
        }
        //REQUIRED contour is detected,then convex hell is found and also convexity defects are found and stored in defects

        if (maxarea>100){
            drawContours(frame, hull, lrgctridx, Scalar(2555, 0, 255), 3, 8, vector<Vec4i>(), 0, Point());

            /// Draw convexityDefects
            for(int j=0; j<defects[lrgctridx].size(); ++j)
            {
                const Vec4i& v = defects[lrgctridx][j];
                float depth = v[3] / 256;
                if (depth > 10) //  filter defects by depth
                {
                    int startidx = v[0]; Point ptStart(Contours[lrgctridx][startidx]);
                    int endidx = v[1]; Point ptEnd(Contours[lrgctridx][endidx]);
                    int faridx = v[2]; Point ptFar(Contours[lrgctridx][faridx]);

                    line(frame, ptStart, ptEnd, Scalar(0, 255, 0), 1);
                    line(frame, ptStart, ptFar, Scalar(0, 255, 0), 1);
                    line(frame, ptEnd, ptFar, Scalar(0, 255, 0), 1);
                    circle(frame, ptFar, 4, Scalar(0, 255, 0), 2);
                }
            }
        }

        imshow("output", frame);
        char key = waitKey(33);
        if (key == 27) break;

    }
}