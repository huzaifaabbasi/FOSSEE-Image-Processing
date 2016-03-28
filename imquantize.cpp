#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#define GRAYSCALE 256

using namespace cv;
using namespace std;

// Name - Huzaifa
// imquantize(Mat A, vector<uchar> thresh) quantizes the image into segments based on the threshold vector
// It assigns values ranging from 0 to size of thresh + 1 for each interval in the thresh vector
// - > Input Parameters  - Mat A is the image file container which has the image data
//						   thresh is the vector based on which the quantizing is done
// - > Output Parameters - Mat final_image is the image after quantizing 

// imquantize(Mat A, vector<uchar> thresh, vector<uchar> values) quantizes the image into segments based on the threshold vector
// It assigns values using the values vector for each interval in the thresh vector
// - > Input Parameters  - Mat A is the image file container which has the image data
//						   thresh is the vector based on which the quantizing is done
//						   values is the vector which is used to assign the values based on the interval	
// - > Output Parameters - Mat final_image is the image after quantizing 


Mat imquantize(Mat A, vector<uchar> thresh)
{
	Mat final_image(A.size(), A.type(), Scalar::all(0));			// Initalizing the final image to all zeroes

	for(int i = 0; i < A.rows; i++)									// It increments the value if the pixel value is greater than the
		for(int j = 0; j < A.cols; j++)								// threshold value. Thus for any interval <i,i+1> the value is i 
			for(int k = 0; k < thresh.size(); k++)
			{
				if(A.at<uchar>(i,j) > thresh[k])
					final_image.at<uchar>(i,j)++;
			}	
	return final_image;		
} 

Mat imquantize(Mat A, vector<uchar> thresh, vector<uchar> values)
{
	Mat final_image(A.size(), A.type(), Scalar::all(values[thresh.size()]));			// Initalizing the final image to thresh[size+1]


	for(int i = 0; i < A.rows; i++)									// It assigns the value of the value vector if the pixel value is
		for(int j = 0; j < A.cols; j++)								// less than the threshold value. Thus the last value will be 
			for(int k = 0; k < thresh.size(); k++)					// for interval <i,i+1> where the pixel value is less than thresh[i+1]
			{														
				if(A.at<uchar>(i,j) < thresh[k])
					final_image.at<uchar>(i,j) = values[k];
			}
	return final_image;		

}

int main(int argc, char** argv)
{
  char* imageName = argv[1];

  Mat image;
  image = imread(imageName, 1);										// Reads the image file

  if(argc != 2 || !image.data)										// Error check: If the file doesn't exist or the parameters to execute are present
  {
    printf("No image data.\n");
    return -1;
  }

  std::vector<uchar> v,val;											// Creating an example vector to to pass as the thresholds
  v.push_back(200);
  val.push_back(0);
  val.push_back(200);
  Mat image_new = imquantize(image,v,val);							// Calling the imquantize function
  imwrite("std_image.jpg", image_new);								// Writing image to the file std_image.jpg

  namedWindow(imageName, CV_WINDOW_AUTOSIZE);						
  namedWindow("Quantize", CV_WINDOW_AUTOSIZE);

  imshow(imageName, image);
  imshow("Quantize", image_new);

  waitKey(0);

  return 0;
}

