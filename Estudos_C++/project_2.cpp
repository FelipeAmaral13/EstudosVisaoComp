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
		Mat imgGray, imgBlur, imgCanny, imgDia, imgErode;

		cvtColor(img, imgGray, COLOR_BGR2RGB);
		//Filter
		GaussianBlur(img, imgBlur, Size(7, 7), 5, 0);
		// Edge detect
		Canny(imgBlur, imgCanny, 1, 240);

		//Morphologic Maths
		Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat kernel_2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(imgCanny, imgDia, kernel);
		erode(imgCanny, imgErode, kernel_2);


		imshow("Image", img);
		imshow("Image Gray", imgGray);
		imshow("Image Blur", imgBlur);
		imshow("Image Canny", imgCanny);
		imshow("Image Dilate", imgDia);
		imshow("Image Erode", imgErode);
		waitKey(1);
	}
}