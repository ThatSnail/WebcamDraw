#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

const enum ModeGenerate {NONE, POSITIVES, TRANSITION, NEGATIVES, DESCRIPTIONS, GEN_POSITIVES, TRAIN_CLASSIFIER, LOAD_CLASSIFIER, DETECT, CLEANUP};
const int POSITIVE_COUNT = 100;
const int NEGATIVE_COUNT = 100;
const int POSITIVE_STEPS = 100;
const int NEGATIVE_STEPS = 100;

int WIDTH;
int HEIGHT;

Point EYE_RECT_SIZE;

Point EYE_RECT_P1;
Point EYE_RECT_P2;

void processText();
void processMode();

ModeGenerate mode = NONE;
int positiveCount = 0;
int negativeCount = 0;
int transition = 0;

Mat frame;

CascadeClassifier cascadeClass;

int main()
{
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
	if(!capture)
	{
		return -1;
	}
	cvNamedWindow("window", CV_WINDOW_AUTOSIZE);

	Mat faceTest = imread("test.png");

	while(true)
	{
		frame = cvQueryFrame(capture);
		cvtColor(frame, frame, CV_BGR2GRAY);
		equalizeHist(frame, frame);

		WIDTH = frame.size.p[1];
		HEIGHT = frame.size.p[0];

		EYE_RECT_SIZE = Point(100, 50);

		EYE_RECT_P1 = Point(WIDTH / 2 - EYE_RECT_SIZE.x / 2, HEIGHT / 2 - EYE_RECT_SIZE.y / 2);
		EYE_RECT_P2 = Point(WIDTH / 2 + EYE_RECT_SIZE.x / 2, HEIGHT / 2 + EYE_RECT_SIZE.y / 2);

		flip(frame, frame, 1);
		rectangle(frame, EYE_RECT_P1, EYE_RECT_P2, Scalar(0, 0, 255));
		
		/*
		if(mode != NEGATIVES)
			faceTest.copyTo(frame(Rect(EYE_RECT_P1, EYE_RECT_P2)));
		*/
		
		
		processText();
		processMode();
		
		if(!frame.data)
		{
			return -1;
		}
		imshow("window", frame);

		if((cvWaitKey(10) & 255) == 32) //SPACE
		{
			if(mode == NONE)
			{
				mode = POSITIVES;
			}
		}
		if((cvWaitKey(10) & 255) == 27) //ESCAPE
			break;
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("window");
	return 0;
}

void processText()
{
	switch(mode)
	{
	case NONE:
		putText(frame, "None", Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		break;
	case POSITIVES:
		{
			std::stringstream ss;
			ss << positiveCount;
			putText(frame, "Running Positives: " + ss.str(), Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		}
		break;
	case TRANSITION:
		{
			std::stringstream ss;
			ss << transition;
			putText(frame, "Close your eyes. Transition: " + ss.str(), Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		}
		break;
	case NEGATIVES:
		{
			std::stringstream ss;
			ss << negativeCount;
			putText(frame, "Running Negatives: " + ss.str(), Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		}
		break;
	case DESCRIPTIONS:
		putText(frame, "Building description files...", Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		break;
	case DETECT:
		putText(frame, "Detecting...", Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		break;
	case CLEANUP:
		putText(frame, "Cleanup...", Point(2, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		break;
	}
}

void processMode()
{
	switch(mode)
	{
	case POSITIVES:
		if(positiveCount < POSITIVE_COUNT)
		{
			for(int i = 0; i < POSITIVE_COUNT/POSITIVE_STEPS; i++)
			{
				if(positiveCount < POSITIVE_COUNT)
				{
					std::stringstream ss;
					ss << positiveCount;
					imwrite("positives/" + ss.str() + ".jpg", frame);
					positiveCount++;
				}
			}
		}
		else
		{
			mode = TRANSITION;
		}
		break;
	case TRANSITION:
		if(transition < 100)
			transition++;
		else
		{
			transition = 0;
			mode = NEGATIVES;
		}
		break;
	case NEGATIVES:
		if(negativeCount < NEGATIVE_COUNT)
		{
			for(int i = 0; i < NEGATIVE_COUNT/NEGATIVE_STEPS; i++)
			{
				if(negativeCount < NEGATIVE_COUNT)
				{
					std::stringstream ss;
					ss << negativeCount;
					imwrite("negatives/" + ss.str() + ".jpg", frame);
					negativeCount++;
				}
				else
					break;
			}
		}
		else
		{
			mode = DESCRIPTIONS;
		}
		break;
	case DESCRIPTIONS:
		{
		std::ofstream file;

		//positives description
		file.open("positives.txt");
		for(int i = 0; i < POSITIVE_COUNT; i++)
		{
			std::stringstream ss0, ss1, ss2, ss3, ss4;
			ss0 << i;
			ss1 << EYE_RECT_P1.x;
			ss2 << EYE_RECT_P1.y;
			ss3 << EYE_RECT_P2.x - EYE_RECT_P1.x;
			ss4 << EYE_RECT_P2.y - EYE_RECT_P1.y;
			std::string s = "positives/" + ss0.str() + ".jpg" + " 1 " + ss1.str() + " " + ss2.str() + " " + ss3.str() + " " + ss4.str();
			file << s << std::endl;
		}
		file.close();

		//negatives description
		file.open("negatives.txt");
		for(int i = 0; i < NEGATIVE_COUNT; i++)
		{
			std::stringstream ss;
			ss << i;
			std::string s = "negatives/" + ss.str() + ".jpg";
			file << s << std::endl;
		}
		file.close();

		mode = GEN_POSITIVES;
		}
		break;
	case GEN_POSITIVES:
		{
			//TODO generate positives
			std::stringstream ss;
			ss << POSITIVE_COUNT;
			system(("opencv_createsamples.exe -info positives.txt -vec positives_vec.txt -bg negatives.txt -num " + ss.str()).c_str());
			mode = TRAIN_CLASSIFIER;
		}
		break;
	case TRAIN_CLASSIFIER:
		{
			//TODO train classifier
			std::stringstream sn;
			sn << NEGATIVE_COUNT;
			system(("opencv_traincascade.exe -data classifier -vec positives_vec.txt -bg negatives.txt -numPos 10 -numNeg " + sn.str() + " -numStages 10").c_str());
			mode = LOAD_CLASSIFIER;
		}
		break;
	case LOAD_CLASSIFIER:
		{
			//TODO load classifer
			if(!cascadeClass.load("cascade.xml"))
			{
				cout << "OH NO" << endl;
			}
			mode = DETECT;
		}
		break;
	case DETECT:
		{
			vector<Rect> vDetects;
			cascadeClass.detectMultiScale(frame, vDetects);
			for(int i = 0; i < vDetects.size(); i++)
			{
				rectangle(frame, vDetects.at(0), Scalar(0, 0, 255));
			}
			vDetects.clear();
		}
		break;
	case CLEANUP:
		{
			positiveCount = 0;
			negativeCount = 0;
			mode = NONE;
		}
		break;
	}
}