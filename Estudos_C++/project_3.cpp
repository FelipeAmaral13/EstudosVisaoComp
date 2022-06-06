#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

	VideoCapture cap(0);
	Mat img;

	while (true) {
		cap.read(img);
		Mat imgResize, imgCrop;

		//resize(img, imgResize, Size(640, 480));
		resize(img, imgResize, Size(), 0.5, 0.5);
		
		//Crop
		Rect roi(100, 100, 300, 250);
		imgCrop = img(roi);

		imshow("Image", imgResize);
		imshow("Image", imgCrop);
		waitKey(1);
	}
}