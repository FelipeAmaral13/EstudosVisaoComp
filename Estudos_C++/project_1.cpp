#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//int main() {
//	string path = "Resources/2.png";
//	Mat img = imread(path);
//
//	imshow("Image", img);
//	waitKey(0);
//}

int main() {

	VideoCapture cap(0);
	Mat img;

	while (true) {
		cap.read(img);

		imshow("Image", img);
		waitKey(1);

	}
	
}