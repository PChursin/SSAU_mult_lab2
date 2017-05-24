#include <stdlib.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace cv;

const int kMenuTabs = 15;
const char* menu[] =
{
	"0  -  Read image", 
	"1  -  Apply linear filter", 
	"2  -  Apply blur(...)", 
	"3  -  Apply medianBlur(...)", 
	"4  -  Apply GaussianBlur(...)", 
	"5  -  Apply erode(...)", 
	"6  -  Apply dilate(...)", 
	"7  -  Apply Sobel(...)", 
	"8  -  Apply Laplacian(...)", 
	"9  -  Apply Canny(...)", 
	"10 -  Apply calcHist(...)", 
	"11 -  Apply equalizeHist(...)",
	"12 -  Apply Extended Morphology(...)",
	"13 -  Apply boxFilter(...)",
	"14 -  Draw primitives"
};
const char* winNames[] =
{
	"Initial image",
	"filter2d",
	"blur",
	"medianBlur",
	"GaussianBlur",
	"erode",
	"dilate",
	"Sobel",
	"Laplacian",
	"Canny",
	"calcHist",
	"equalizeHist",
	"Extended morphology",
	"Box filter"
};
const int maxFileNameLen = 1000;
const int escCode = 27;

void printMenu()
{
	printf("Menu items:\n"); 
	for (int i = 0; i < kMenuTabs; i++)
	{
		printf("\t%s\n", menu[i]); 
	}
	printf("\n"); 
}
void loadImage(Mat &srcImg)
{
	char fileName[maxFileNameLen];
	do
	{
		printf("Input full file name: ");
		scanf("%s", &fileName);
		srcImg = imread(fileName, 1);
	} while (srcImg.data == 0);
	printf("The image was succesfully read\n\n"); 
}
void chooseMenuTab(int &activeMenuTab, Mat &srcImg)
{
	int tabIdx;
	while (true)
	{
		// print menu items 
		printMenu();
		// get menu item identifier to apply operation 
		printf("Input item identifier to apply operation: "); 
		scanf("%d", &tabIdx);
		if (tabIdx == 0)
		{
			// read image 
			loadImage(srcImg);
		}
		else if (tabIdx >= 1 && tabIdx < kMenuTabs && srcImg.data == 0)
		{
			printf("The image should be read to apply operation!\n"); 
			loadImage(srcImg);
		}
		else if (tabIdx >= 1 && tabIdx < kMenuTabs)
		{
			activeMenuTab = tabIdx;
			break;
		}
	}
}
int applyOperation(const Mat &src, const int operationIdx)
{
	char key = -1;
	Mat dst;
	switch (operationIdx)
	{
		case 1:
		{
			const float kernelData[] = {-0.1f, 0.2f,-0.1f,0.2f, 3.0f, 0.2f,-0.1f, 0.2f,-0.1f };
			const Mat kernel(3, 3, CV_32FC1, (float *)kernelData);
			filter2D(src, dst, -1, kernel);
			break;
		}
		case 2:
		{
			printf("Enter blur dimensions: ");
			int x, y;
			scanf("%d %d", &x, &y);
			printf("\n");
			blur(src, dst, Size(x, y));
			break;
		}
		case 3:
		{
			printf("Enter blur size: ");
			int x;
			scanf("%d", &x);
			printf("\n");
			medianBlur(src, dst, x);
			break;
		}
		case 4:
		{
			printf("Enter blur dimensions: ");
			int x, y;
			scanf("%d %d", &x, &y);
			printf("\n");
			printf("Enter sigmaX: ");
			int sx;
			scanf("%d", &sx);
			printf("\n");
			GaussianBlur(src, dst, Size(x, y), sx);
			break;
		}
		case 5:
		{
			printf("Enter erode iterations: ");
			int it;
			scanf("%d", &it);
			erode(src, dst, Mat(), Point(-1, -1), it, 0, morphologyDefaultBorderValue());
			break;
		}
		case 6:
		{
			printf("Enter dilate iterations: ");
			int it;
			scanf("%d", &it);
			dilate(src, dst, Mat(), Point(-1, -1), it, 0, morphologyDefaultBorderValue());
			break;
		}
		case 7:
		{
			Mat gray, xGrad, yGrad, xGradAbs, yGradAbs;
			printf("Type 1 to blur image first: ");
			int blur;
			scanf("%d", &blur);
			if (blur == 1)
				GaussianBlur(src, gray, Size(3, 3), 0);
			else
				src.copyTo(gray);
			cvtColor(gray, gray, CV_RGB2GRAY);
			printf("Enter x and y orders: ");
			int x, y;
			scanf("%d %d", &x, &y);
			Sobel(gray, xGrad, CV_16S, x, 0);
			Sobel(gray, yGrad, CV_16S, 0, y);
			convertScaleAbs(xGrad, xGradAbs);
			convertScaleAbs(yGrad, yGradAbs);
			addWeighted(xGradAbs, 0.5, yGradAbs, 0.5, 0, dst);
			break;
		}
		case 8:
		{
			Mat gray, lap;
			printf("Type 1 to blur image first: ");
			int blur;
			scanf("%d", &blur);
			if (blur == 1)
				GaussianBlur(src, gray, Size(3, 3), 0);
			else
				src.copyTo(gray);
			cvtColor(gray, gray, CV_RGB2GRAY);
			Laplacian(gray, lap, CV_16S);
			convertScaleAbs(lap, dst);
			break;
		}
		case 9:
		{
			Mat gray;
			printf("Type 1 to blur image first: ");
			int blur;
			scanf("%d", &blur);
			if (blur == 1)
				GaussianBlur(src, gray, Size(3, 3), 0);
			else
				src.copyTo(gray);
			printf("Enter lowThreshold: ");
			int t;
			scanf("%d", &t);
			cvtColor(gray, gray, CV_RGB2GRAY);
			Canny(gray, dst, (double)t, 3.0*t);
			break;
		}
		case 10:
		{
			Mat bgrChannels[3], bHist, gHist, rHist;
			int kBins = 256;
			float range[] = { 0.0f, 256.0f };
			const float* histRange = { range };
			bool uniform = true;
			bool accumulate = false;
			int hWidth = 512, hHeight = 400;
			int binWidth = cvRound((double) hWidth / kBins);
			Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };
			split(src, bgrChannels);
			calcHist(&bgrChannels[0], 1, 0, Mat(), bHist, 1, &kBins, &histRange, uniform, accumulate);
			calcHist(&bgrChannels[1], 1, 0, Mat(), gHist, 1, &kBins, &histRange, uniform, accumulate);
			calcHist(&bgrChannels[2], 1, 0, Mat(), rHist, 1, &kBins, &histRange, uniform, accumulate);
			Mat hist = Mat(hHeight, hWidth, CV_8UC3, Scalar(0, 0, 0));
			normalize(bHist, bHist, 0, hist.rows, NORM_MINMAX, -1, Mat());
			normalize(gHist, gHist, 0, hist.rows, NORM_MINMAX, -1, Mat());
			normalize(rHist, rHist, 0, hist.rows, NORM_MINMAX, -1, Mat());
			for (int i = 1; i < kBins; i++) {
				line(hist, Point(binWidth*(i - 1), hHeight - cvRound(bHist.at<float>(i - 1))),
					Point(binWidth * i, hHeight - cvRound(bHist.at<float>(i))), colors[0], 2, 8, 0);
				line(hist, Point(binWidth*(i - 1), hHeight - cvRound(gHist.at<float>(i - 1))),
					Point(binWidth * i, hHeight - cvRound(gHist.at<float>(i))), colors[1], 2, 8, 0);
				line(hist, Point(binWidth*(i - 1), hHeight - cvRound(rHist.at<float>(i - 1))),
					Point(binWidth * i, hHeight - cvRound(rHist.at<float>(i))), colors[2], 2, 8, 0);
			}
			hist.copyTo(dst);
			break;
		}
		case 11:
		{
			Mat gray;
			cvtColor(src, gray, CV_RGB2GRAY);
			equalizeHist(gray, dst);
			namedWindow("gray", 1);
			imshow("gray", gray);
			break;
		}
		case 12:
		{
			int ch = 0;
			while (ch < 1 || ch > 5) {
				printf("Choose morphology kind:\n1) OPEN\n2) CLOSE\n3) GRADIENT\n4) TOPHAT\n5) BLACKHAT\nYour choice ---> ");
				scanf("%d", &ch);
			}
			int morph;
			switch (ch) {
			case 1:
				morph = MORPH_OPEN;
				break;
			case 2:
				morph = MORPH_CLOSE;
				break;
			case 3:
				morph = MORPH_GRADIENT;
				break;
			case 4:
				morph = MORPH_TOPHAT;
				break;
			case 5:
				morph = MORPH_BLACKHAT;
				break;
			}
			Mat element;
			morphologyEx(src, dst, morph, element);
			break;
		}
		case 13:
		{
			printf("Enter blur dimensions: ");
			int x, y;
			scanf("%d %d", &x, &y);
			printf("\n");
			boxFilter(src, dst, -1, Size(x, y));
			break;
		}
		case 14:
		{
			namedWindow(winNames[0], 1);
			imshow(winNames[0], src);
			waitKey();
			int ch = 1;
			//int it = 0;
			std::vector<Mat> matV;
			Mat first;
			src.copyTo(first);
			matV.push_back(first);
			while (ch - 7)
			{
				printf("Choose action:\n");
				printf("1) Draw line...\n2) Draw circle...\n3) Draw ellipse...\n");
				printf("4) Draw rectangle...\n5) Draw poly...\n6) Undo\n");
				printf("7) Exit drawing mode\n Choice --> ");
				scanf("%d", &ch);
				switch (ch) {
				case 1:
				{
					int x, y;
					printf("Enter first point (x y): ");
					scanf("%d %d", &x, &y);
					Point f(x, y);
					printf("Enter second point (x y): ");
					scanf("%d %d", &x, &y);
					Point s(x, y);
					int thick;
					printf("Enter thickness: ");
					scanf("%d", &thick);
					int r, g, b;
					printf("Enter color components (r, g, b): ");
					scanf("%d %d %d", &r, &g, &b);
					Scalar col(b, g, r);
					Mat temp;
					matV.back().copyTo(temp);
					line(temp, f, s, col, thick, LINE_8);
					matV.push_back(temp);
					break;
				}
				case 2:
				{
					int x, y;
					printf("Enter center point (x y): ");
					scanf("%d %d", &x, &y);
					Point f(x, y);
					printf("Enter radius: ");
					int rad;
					scanf("%d", &rad);
					int thick;
					printf("Enter thickness (or 0 to fill): ");
					scanf("%d", &thick);
					int r, g, b;
					printf("Enter color components (r, g, b): ");
					scanf("%d %d %d", &r, &g, &b);
					Scalar col(b, g, r);
					Mat temp;
					matV.back().copyTo(temp);
					circle(temp, f, rad, col, (thick <= 0 ? FILLED : thick), LINE_8);
					matV.push_back(temp);
					break;
				}
				case 3:
				{
					int x, y;
					printf("Enter center of bounding box (x y): ");
					scanf("%d %d", &x, &y);
					Point c(x, y);
					printf("Enter size of the box (w h): ");
					int w, h;
					scanf("%d %d", &w, &h);
					printf("Enter angle of the box: ");
					float ang;
					scanf("%e", &ang);
					int thick;
					printf("Enter thickness (or 0 to fill): ");
					scanf("%d", &thick);
					int r, g, b;
					printf("Enter color components (r, g, b): ");
					scanf("%d %d %d", &r, &g, &b);
					Scalar col(b, g, r);
					Mat temp;
					matV.back().copyTo(temp);
					ellipse(temp, RotatedRect(c, Size(w, h), ang), col, (thick <= 0 ? FILLED : thick), LINE_8);
					matV.push_back(temp);
					break;
				}
				case 4:
				{
					int x, y;
					printf("Enter upper-left point (x y): ");
					scanf("%d %d", &x, &y);
					Point f(x, y);
					printf("Enter opposite point (x y): ");
					scanf("%d %d", &x, &y);
					Point s(x, y);
					int thick;
					printf("Enter thickness (or 0 to fill): ");
					scanf("%d", &thick);
					int r, g, b;
					printf("Enter color components (r, g, b): ");
					scanf("%d %d %d", &r, &g, &b);
					Scalar col(b, g, r);
					Mat temp;
					matV.back().copyTo(temp);
					rectangle(temp, f, s, col, (thick <= 0 ? FILLED : thick), LINE_8);
					matV.push_back(temp);
					break;
				}
				case 5:
				{
					int npts;
					printf("Enter number of points: ");
					scanf("%d", &npts);
					Point * pts = new Point[npts+1];
					for (int i = 0; i < npts; i++) {
						int x, y;
						printf("Enter point (x y): ");
						scanf("%d %d", &x, &y);
						pts[i] = Point(x, y);
					}
					//pts[npts] = pts[0];
					int r, g, b;
					printf("Enter color components (r, g, b): ");
					scanf("%d %d %d", &r, &g, &b);
					Scalar col(b, g, r);
					Mat temp;
					matV.back().copyTo(temp);
					fillConvexPoly(temp, pts, npts, col);
					matV.push_back(temp);
					break;
				}
				case 6:
				{
					if (matV.size()-1)
						matV.pop_back();
					else
						printf("There is nothing to undo!");
					break;
				}
				default:
					break;
				}
				namedWindow("Drawing", 1);
				imshow("Drawing", matV.back());
				waitKey();
			}
		}
		default:
			break;
	}
	if (operationIdx != 14) {
		namedWindow(winNames[0], 1);
		imshow(winNames[0], src);
		namedWindow(winNames[operationIdx]);
		imshow(winNames[operationIdx], dst);
	}
	return 0;
}

int main(int argc, char** argv)
{	
	Mat srcImg;
	char ans;
	int activeMenuTab = -1;
	do
	{
		// вызов функции выбора пункта меню 
		chooseMenuTab(activeMenuTab, srcImg);
		// применение операций 
		applyOperation(srcImg, activeMenuTab);
		// вопрос о необходимости продолжения 
		printf("Do you want to continue? ESC - exit\n");
		// ожидание нажатия клавиши 
		ans = waitKey();
	} while (ans != escCode);
	destroyAllWindows(); // закрытие всех окон 
	srcImg.release(); // освобожение памяти */
	return 0;
}