#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#define GRAYSCALE 256

using namespace cv;
using namespace std;


// Name - Huzaifa
// lookupTables() builds lookup tables which will be used in the Otsu's algorithm
// - > Input Parameters - nb[i][j] This is the sum of the probabilties from i to j 
//                        mu[i][j] This is the sum of product of k with the probability of k
//                        sigma[i][j] This is the sub-variance of the class from i to j
//                        prob[i] is the probability of the pixel value i
// - > Output Parameters -None

// findMaxSigma() finds the maximum between class variance for each possible class partition
// - > Input Parameters - level gives the number of classes the partition must have
//                        sigma[i][j] This is the sub-variance of the class from i to j
//                        t[] is the threshold values returned from the function
// - > Output Parameters -maxSigma is the maximum between class variance

// multithresh( Mat img, int N) calls the above two functions to calculate the threshold values
// -> Input Parameters -  img is the Mat image whose threshold is to be calculated
//                        N is the number of threshold values that it is supposed to determine
// -> Output Parameters - v is the vector of integers which contains the threshold values 

// Sources - http://www.iis.sinica.edu.tw/JISE/2001/200109_01.pdf               





void LookupTables(double nb[GRAYSCALE][GRAYSCALE], double mu[GRAYSCALE][GRAYSCALE], double sigma[GRAYSCALE][GRAYSCALE], double prob[GRAYSCALE])   // initialize
 {
    for (int j=0; j < GRAYSCALE; j++)              // Initializing the lookup table
      for (int i=0; i < GRAYSCALE; ++i)
      {
		nb[i][j] = 0.0;
		mu[i][j] = 0.0;
		sigma[i][j] = 0.0;
      }

    for (int i=0; i < GRAYSCALE; ++i)              // Determining the diagonal entries              
    {
      nb[i][i] = prob[i];
      mu[i][i] = (double) (i)*prob[i];
    }
    
    for (int i=0; i < GRAYSCALE-1; ++i)            // Determining the first row (row 0 is the first row)
    {
      nb[0][i+1] = nb[0][i] + prob[i+1];
      mu[0][i+1] = mu[0][i] + (double) ((i+1))*prob[i+1];
    }
    
    for (int i=1; i < GRAYSCALE; i++)              // using row 0 to calculate others
      for (int j=i; j < GRAYSCALE; j++)
      {
	       nb[i][j] = nb[0][j] - nb[0][i-1];
	       mu[i][j] = mu[0][j] - mu[0][i-1];
      }
    
    for (int i=0; i < GRAYSCALE; ++i)              // Determining the sub-variance for each class
      for (int j=i+1; j < GRAYSCALE; j++)
      {
	       if (nb[i][j] != 0)
	         sigma[i][j] = (mu[i][j]*mu[i][j])/nb[i][j];
	       else
	         sigma[i][j] = 0.0;
      }

  }

double findMaxSigma(int level, double sigma[GRAYSCALE][GRAYSCALE], int t[])
  {
    t[0] = 0;
    double maxSig= 0.0;                           // Maximum between class variance
    switch(level)
    {
    case 2:                                       // For 1 threshold value
    	for (int i= 0; i < GRAYSCALE-level + 2; i++) // t1
    	{
			double Sq = sigma[0][i] + sigma[i+1][255];
			if (maxSig < Sq)
			{
	  			t[0] = i;
	  			maxSig = Sq;
			}
   		} 
   		break;
    case 3:                                       // For 2 threshold values
      	for (int i= 0; i < GRAYSCALE-level + 2; i++) // t1
			for (int j = i+1; j < GRAYSCALE-level + 3; j++) // t2
			{
	  			double Sq = sigma[0][i] + sigma[i+1][j] + sigma[j+1][255];
	  			if (maxSig < Sq)
	  			{
	    			t[0] = i;
	    			t[1] = j;
	    			maxSig = Sq;
	  			}
			} 
      	break;
    case 4:                                       // For 3 threshold values
      	for (int i= 0; i < GRAYSCALE-level + 2; i++) // t1
			for (int j = i+1; j < GRAYSCALE-level + 3; j++) // t2
	  			for (int k = j+1; k < GRAYSCALE-level + 4; k++) // t3
	  			{
	    			double Sq = sigma[0][i] + sigma[i+1][j] + sigma[j+1][k] + sigma[k+1][255];
	    			if (maxSig < Sq)
	    			{
	      				t[0] = i;
	      				t[1] = j;
	      				t[2] = k;
	      				maxSig = Sq;
	    			}
	  			} 
      	break;
    case 5:                                       // For 4 threshold values 
      	for (int i= 0; i < GRAYSCALE-level + 2; i++) // t1
			for (int j = i+1; j < GRAYSCALE-level + 3; j++) // t2
	  			for (int k = j+1; k < GRAYSCALE-level + 4; k++) // t3
	    			for (int m = k+1; m < GRAYSCALE-level + 5; m++) // t4
	  				{
	    				double Sq = sigma[0][i] + sigma[i+1][j] + sigma[j+1][k] + sigma[k+1][m] + sigma[m+1][255];
	    				if (maxSig < Sq)
	    				{
					      t[0] = i;
					      t[1] = j;
					      t[2] = k;
					      t[3] = m;
					      maxSig = Sq;
	    				}
	  				} 
      	break;
    }
    return maxSig; 
  }

vector<int> multithresh(Mat img, int N)
{
    vector<int> v;
    Mat img2 = Mat(img.rows, img.cols, CV_8UC1);
    if( img.empty())                              // Error check: If image file exists or not
    {
        cout<< "File could not be found" <<endl;
        return v;
    }

    if( img.channels() > 1)                       // Converting RGB image to Gray image
           cvtColor(img, img2, CV_BGR2GRAY);
    else
        img2 = img;

    int hist[GRAYSCALE];                          // Histogram of the pixel values
    double prob[GRAYSCALE];                       // Probability of pixel value

    for(int i = 0 ; i < GRAYSCALE; i++)           // initializes the histogram to zero
         hist[i] = 0;

    for(int i = 0; i < img2.rows; i++)            // Creates the histogram 
        for(int j = 0; j < img2.cols; j++)
            hist[ img2.at<uchar>(i,j) ]++;

    double size = img2.rows * img2.cols;          //Total number of pixels

    for(int i = 0; i < GRAYSCALE; i++)            // Creating the probabilty array
    	prob[i] = (double) (hist[i]/size);    

    int thresh[5];

    double nb[GRAYSCALE][GRAYSCALE], mu[GRAYSCALE][GRAYSCALE], sigma[GRAYSCALE][GRAYSCALE];   //

    LookupTables(nb,mu,sigma,prob);     // Creating the lookup tables 
    findMaxSigma(N+1, sigma, thresh);   // Using the lookup tables to find the maximum between class variance

    v = vector<int>(thresh, thresh + sizeof thresh / sizeof thresh[0]);   // creating vector from the array
    return v;
}

vector<int> multithresh(Mat img)
{
  return multithresh(img, 1);
}

int main(int argc, char** argv)
{
  char* imageName = argv[1];

  Mat image;
  image = imread(imageName, 1);     // Reads the image file

  if(argc != 2 || !image.data)      // Error check: If the file doesn't exist or the parameters to execute are present
  {
    printf("No image data.\n");
    return -1;
  }

  vector<int> v = multithresh(image,3);    // Calls the multithresh function and assigns the threshholds to v
  for(int i = 0; i < 3; i++)
    cout<<v[i]<<" ";
  cout<<endl;
    waitKey(0);

  return 0;
}        

