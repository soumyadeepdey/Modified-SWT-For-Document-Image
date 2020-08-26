 

//~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*
// Program Name :       Stroke Width Transform(SWT)
// 
// Project :  		DRD
// Author : 		Soumyadeep Dey
// Creation Date : 	MAR  -2014.  Rights Reserved
//~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*~^~*

 
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <sys/stat.h>
#include <iostream>


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"


#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;


#define PI 3.14159265

struct BeanStucture{
  int BeanNumber;
  int MaxElement;
  int LowerBound;
  int UpperBound;
  int middle; 
};

struct Ray {
        Point2i p;
        Point2i q;
        vector<Point2i> points;
	int dist;
	void CalcEcluiDist()
	{
	  dist =(int) sqrt( ((p.x - q.x)*(p.x-q.x)) + ((p.y-q.y)*(p.y-q.y)) );
	}
};


 Mat src,dts;


 RNG rng(12345);

 

/**
 * @function validate
 * @param : input param: co-ordinate position(i,j) and maximum-limit(t) row, col
 * @brief : co-ordinate position(i,j) to be check whether it is within given row and col
 * @return : 1 if it belong to particular region
 *           0 if not belong within that particular row and col
 */


int validate(int i, int j, int row, int col)
{
  
  if(i<0 || i>=row || j<0 || j>=col)
    return 0;
  else
    return 1;
}

 
/*-------------------------------------------------MAKE DIRECTORY FUNCTION-------------------------------------------*/
/**
 * @function makedir
 * @param input a character string
 * @brief it create a directry of given character string
 */
void makedir(char *name)
{
	int status;
	status=mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
 
/*-------------------------------------------------cut string upto( .)-------------------------------------------*/

/**
 * @function input_image_name_cut
 * @param : input param: input-name to be cut upto '.'
 * @return : input name upto '.' 
 *
 */


char* input_image_name_cut(char *s) 
{
                 
                     int i,j; 
		     
		     char *substring;
		     
		     substring = (char *)malloc(2001 * sizeof(char));
              
                 for(i=0; i <= strlen(s)-1; i++)
                      {
			
                       if (s[i]!='.' )
		        substring[i] = s[i];
		       else
			 break;
                       }
                       substring[i] = '\0';
                 
                     printf(" %s\n", substring);
		 
		 return(substring);
		     
		     
                      
      }



/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function CreateNameIntoFolder
 * @param  input Foldername, desired name 
 * @return : name within the desired folder
 *
 */

char* CreateNameIntoFolder(char *foldername, char *desiredname )
{
  char *name,*output,*tempname, *tempname1;
  output = (char *) malloc ( 2001 * sizeof(char));
  if(output == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  strcpy(output,foldername);
 
  tempname = (char *) malloc ( 2001 * sizeof(char));
  if(tempname == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  tempname = "/";
  strcat(output,tempname);
  
  tempname1 = (char *) malloc ( 2001 * sizeof(char));
  if(tempname1 == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  strcpy(tempname1,output);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  strcpy(name,tempname1);
  
  strcat(name,desiredname);
  
  return(name);
  
  
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/

 
/*------------------------------------------------------BINARIZATION-------------------------------------------------------------------*/


// parameters for binarization

int binary_threshold_value = 211;

  /**
   * @param :thereshold_type
   * 
   * 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
int threshold_type = 0;
int const maximum_value = 255;
int const maximum_type = 4;
int const maximum_BINARY_value = 255;
//int const blockSize=251;
int const blockSize=151;
Mat TempGray,TempBinary;



void BinaryThreshold( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( TempGray, TempBinary,  binary_threshold_value, maximum_BINARY_value,threshold_type );
  imshow("BinaryThresholding",TempBinary);
}


/**
 * @function binarization
 * @param input an image in Mat format and type for binarization
 * @brief type = 1 for adaptive
 * @brief type = 2 for otsu
 * @brief type = 3 for Normal Threshold by GUI
 * @brief type = 4 for Normal Threshold by fixed value
 * @return Return a binary image of Mat data-type 
  */

Mat binarization(Mat image, int type)
{
/**
 * @param type 
 * type = 1 for adaptive;
 * type = 2 for Otsu
 * type = 3 for Normal Threshold by GUI   
**/
       
	// Convert the image to Gray
  	printf("In Binarization\n");
  	
  	Mat gray,binary;
	threshold_type = 0;
	
	cvtColor(image, gray, CV_BGR2GRAY);
	
	if(type == 1)
	{
	  adaptiveThreshold(  gray, binary, maximum_BINARY_value, ADAPTIVE_THRESH_MEAN_C,  threshold_type,  blockSize, 10);
	  return (binary);
	}
	
	// Otsu Thresholding
	if(type == 2)
	{
	  double val = threshold( gray, binary, 100, maximum_BINARY_value, cv::THRESH_OTSU | cv::THRESH_BINARY);
	  printf("threshold value is %lf\n",val);
	  return (binary);
	}
	
	//GUI Threshold
	if(type == 3)
	{
	  gray.copyTo(TempGray);
	  /// Create a window to display results
	    namedWindow( "BinaryThresholding", CV_WINDOW_KEEPRATIO );
	    
	    createTrackbar( "Value",
			    "BinaryThresholding", & binary_threshold_value,
			    maximum_BINARY_value, BinaryThreshold );

	    /// Call the function to initialize
	    BinaryThreshold( 0, 0 );
	    waitKey(0);
	    printf("threshold value is %d\n",binary_threshold_value);
	    destroyWindow("BinaryThresholding");
	    TempBinary.copyTo(binary);
	    return (binary);
	}
	
	// Fixed Threshold
	if(type == 4)
	{
	  binary_threshold_value = 211;
	  threshold( gray, binary,  binary_threshold_value, maximum_BINARY_value,threshold_type );
	  return (binary);
	}
	
	
	  
	

}


/*------------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function foreground_masked_image
 * @param input an image in Mat format
 * @brief convert an input image into a uniform background image
 * @brief masked the foreground pixels and make the background pixel uniform
 * @return Return a uniform background image of Mat data-type 
 */


Mat foreground_masked_image(Mat image)
{
  Mat binary,uniform;
  
  binary = binarization(image,2);
  image.copyTo(uniform);
  int row = image.rows;
  int col = image.cols;
  
  for(int i =0;i<row;i++)
  {
    for(int j=0;j<col;j++)
    {
      if(binary.data[i*col+j] == 255)
      {
	for(int k=0;k<3;k++)
	  uniform.data[(i*col+j)*3+k]=255;
      }
    }
  }
  
  return(uniform);
  
}


/*------------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function NumberofForegroundPixel
 * @param input an image in Mat format
 * @brief It count number of foreground pixel in the given image
 * @return Return a integer which gives the count of number of foreground pixel 
 */


int NumberofForegroundPixel(Mat image)
{
  Mat binary;
  int row = image.rows;
  int col = image.cols;
  int pixel_count=0;
  binary = binarization(image,2);
  
  for(int i =0;i<row;i++)
  {
    for(int j=0;j<col;j++)
    {
      if(binary.data[i*col+j] == 0)
	pixel_count = pixel_count + 1;
    }
  }
  
  return(pixel_count);
}


/*-------------------------------------------------------------------------------------------------------------------------*/


/**
 * @Function: PointRectangleTest
 * @brief : Take 1 rectangle and a Point as input 
 * @brief : Test whether the Given Point is inside the Given Rectangle or Inside
 * @return : 	0: Point is Outside of Rectangle
 * 		1: Point is inside the given Rectangle
 * */

int PointRectangleTest(Rect GivenRect, Point GivenPoint)
{
  Point tl,br;
  tl = GivenRect.tl();
  br = GivenRect.br();
  int flag;
  /*
  if((GivenPoint.x>=tl.x && GivenPoint.x<=br.x) && (GivenPoint.y<=tl.y && GivenPoint.y>=br.y))
  {
    flag = 1;
    printf("point inside\n");
    return(flag);
  }
  */
  if((GivenPoint.x>=tl.x && GivenPoint.x<=br.x) && (GivenPoint.y>=tl.y && GivenPoint.y<=br.y))
  {
    flag = 1;
    //printf("point inside\n");
    return(flag);
  }
  else
  {
    flag = 0;
    return(flag);
  } 
}

/**
 * @Function: FindOverlappingRectangles
 * @brief : Take 2 rectangle as an input 
 * @return : 	0: Rect 1 and Rect 2 are disjoint
 * 		1: Rect 1 is inside Rect 2
 * 	    	2: Rect 2 is inside Rect 1
 * 	    	3: Rect 1 and 2 are partly overlapped
 * 		
 * 
 * */

int FindOverlappingRectangles(Rect first, Rect second)
{

 Point tl1,tl2,br1,br2;
 
 int flag;

 tl1 = first.tl();
 tl2 = second.tl();
 br1 = first.br();
 br2 = second.br();
 
 if(PointRectangleTest(first,tl2) == 0 && PointRectangleTest(first,br2) == 0)
 {
   flag = 0;
   //return (flag);
 }
 
 if(PointRectangleTest(first,tl2) == 1 || PointRectangleTest(first,br2) ==1 || PointRectangleTest(second,tl1) == 1 || PointRectangleTest(second,br1) == 1)
 {
   flag = 3;
   ///return (flag);
 }
 if(PointRectangleTest(first,tl2) == 1 && PointRectangleTest(first,br2) == 1)
 {
   flag = 2;
   //return (flag);
 }
 if(PointRectangleTest(second,tl1) == 1 && PointRectangleTest(second,br1) == 1)
 {
   flag = 1;
   //return (flag);
 }
 return (flag);
 
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/*----------------------------------------------MORPHOLOGICAL OPERATIONS----------------------------------------------------------------------*/



/*-------------------------------------------------------EROTION WITH 4 NEIGHBOURHOOD-------------------------------------------------------------*/


/**
 * @function erosion
 * @param input an image(binary) in Mat format
 * @brief it erode an image with a square mask of 3x3
 * @return Return eroded image of Mat data-type
 */

Mat erosion(Mat image)
{
	int row = image.rows;
	int col = image.cols;
	int i,j;
	Mat tempimage;
	image.copyTo(tempimage);
	for(i=0;i<row;i++)
	{
	  for(j=0;j<col;j++)
	    tempimage.data[i*col+j] = 255;
	}
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(image.data[i*col+j]==0)
			{
				if(i-1<0||i+1>=row||j-1<0||j+1>=col)
					tempimage.data[i*col+j]=255;
				else if(image.data[(i-1)*col+j]==0&&image.data[(i+1)*col+j]==0&&image.data[i*col+(j-1)]==0&&image.data[i*col+(j+1)]==0)
					tempimage.data[i*col+j]=0;
				else
					tempimage.data[i*col+j]=255;
			}
			else
				tempimage.data[i*col+j]=255;
		}
	}

	return (tempimage);
	
		
}

/*-------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------- BOUNDARY EXTRACTION--------------------------------------------------*/


/**
 * @function boundaryextraction
 * @param input an image(binary) in Mat format
 * @brief it find the boundary of the input image 
 * @return Return boundary of input image(binary in nature)
 */

Mat boundaryextraction(Mat image)
{
	
	int i,j,k;

	Mat erodedimage;
	Mat extractedimage;
	
	image.copyTo(erodedimage);
	image.copyTo(extractedimage);
	int row,col;
	
	row = image.rows;
	col = image.cols;
	
	for(i=0;i<row;i++)
	{
	  for(j=0;j<col;j++)
	  {
	    erodedimage.data[i*col+j] = 255;
	    extractedimage.data[i*col+j] = 255;
	  }
	}
	
	erodedimage=erosion(image);
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(image.data[i*col+j]==erodedimage.data[i*col+j])
				extractedimage.data[i*col+j]=255;
			else
				extractedimage.data[i*col+j]=0;
		}
	}

	return(extractedimage);
	
	
}




/**  @function Erosion  
 * @param input 
 * element type
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input erosion Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Eroded Image of the input image with given kernel type and size 
 * @return Return Eroded image of input image
 */
Mat Erosion( int erosion_elem, int erosion_size, Mat image)
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  Mat ErodedImage;
  /// Apply the erosion operation
  erode( image, ErodedImage, element );
  return(ErodedImage);
  
}


/**  @function Dilation  
 * @param input 
 * element type
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input Dilation Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Dilated Image of the input image with given kernel type and size 
 * @return Return Dilateded image of input image
 */
Mat Dilation( int dilation_elem, int dilation_size, Mat image )
{
  Mat DilatedImage;
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( image, DilatedImage, element );
  return(DilatedImage);
  
}


/**  @function Open  
 * @param input 
 * element type   
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input element Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Open Image of the input image with given kernel type and size 
 * @return Return Open image of input image
 */
Mat Open(int open_elem, int open_size, Mat image)
{
  Mat OpenImage;
  
  int open_type;
  if( open_elem == 0 ){ open_type = MORPH_RECT; }
  else if( open_elem == 1 ){ open_type = MORPH_CROSS; }
  else if( open_elem == 2) { open_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( open_type,
                                       Size( 2*open_size + 1, 2*open_size+1 ),
                                       Point( open_size, open_size ) );
  Mat ErodedImage;
  erode(image, ErodedImage, element);
  dilate(ErodedImage, OpenImage, element);
  //ErodedImage = Erosion(open_elem,open_size, image);
  //OpenImage = Dilation(open_elem,open_size, ErodedImage);
  return(OpenImage);
}


/**  @function Close  
 * @param input  
 * element type   
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input element Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Close Image of the input image with given kernel type and size 
 * @return Return Close image of input image(binary in nature)
 */
Mat Close(int close_elem, int close_size, Mat image)
{
  Mat CloseImage;
  int close_type;
  if( close_elem == 0 ){ close_type = MORPH_RECT; }
  else if( close_elem == 1 ){ close_type = MORPH_CROSS; }
  else if( close_elem == 2) { close_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( close_type,
                                       Size( 2*close_size + 1, 2*close_size+1 ),
                                       Point( close_size, close_size ) );
  Mat DilatedImage;
  dilate( image, DilatedImage, element );
  erode(DilatedImage, CloseImage, element);
 // DilatedImage = Dilation(close_elem, close_size, image);
 // CloseImage = Erosion(close_elem, close_size, DilatedImage);
  return(CloseImage);
}


/*-------------------------------------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------SMOOTHING OPERATIONS----------------------------------------------------------------*/


/*----------------------------------------------------------HORIZONTAL SMOOTHING----------------------------------------------------------------*/


/**
 * @function horizontal_smoothing
 * @param input an image(binary) in Mat format and integer value whitespace that need to be smoothen or filled up
 * @brief with this function from a foreground position , next whitespace number of pixel is filled in horizontal direction
 * @brief it produce smoothed image of the input image in horizontal direction by filling up with foreground with whitespace amount
 * @return Return horizontally smoothed image of input image(binary in nature)
 */

Mat horizontal_smoothing(Mat image, int whitespace)
{
	int i,j,k;
	int row = image.rows;
	int col = image.cols;
	
	Mat hsmoothedimage;
	
        image.copyTo(hsmoothedimage);
	
	for(i=0;i<image.rows;i++)
	{
		for(j=0;j<image.cols;j++)
		{			
			if(image.data[(i*image.cols)+j]==0)
			{
				for(k=j+1;k<(j+1+whitespace);k++)
				{
					if(k<image.cols)
					{					
						hsmoothedimage.data[(i*image.cols)+k]=0;
					}
					else 
						break;
					
				}
			}
		}
	}

	
	return(hsmoothedimage);
	
}


/**
 * @function horizontal_gapfilling
 * @param input an image(binary) in Mat format and integer value whitespace that need to be smoothen or filled up
* @brief with this function gap btween two foreground pixel is filled only if the gap between two foreground pixel in horizontal direction has a gap less than or equal to whitespace
 * @brief it produce smoothed image of the input image in horizontal direction by filling up with foreground with whitespace amount
 * @return Return gap-filled image of input image(binary in nature)
 */

Mat horizontal_gapfilling(Mat image, int whitespace)
{
	int i,j,k,l;
	
	Mat hgapfilled;
	
        image.copyTo(hgapfilled);
	
	for(i=0;i<image.rows;i++)
	{
	  for(j=0;j<image.cols;j++)
	  {
	    if(image.data[(i*image.cols)+j]==0)
	    {
	      if( (j+whitespace)<hgapfilled.cols)
	      {
		for(k=j+whitespace;k>=j+1;k--)
		{
		  if(image.data[(i*image.cols)+k]==0)
		  {
		    for(l=k;l>=j+1;l--)
		      hgapfilled.data[(i*hgapfilled.cols)+l]=0;
		    break;
		  }
		}
	      }
	      else
		break;
	    }
	  }
	}
	
	return (hgapfilled);
}

/*---------------------------------------------------------VERTICAL SMOOTHING----------------------------------------------------------------*/


/**
 * @function vertical_smoothing
 * @param input an image(binary) in Mat format and integer value whitespace that need to be smoothen or filled up
 * @brief with this function from a foreground position , next whitespace number of pixel is filled
 * @brief it produce smoothed image of the input image in vertical direction by filling up with foreground with whitespace amount
 * @return Return vertically smoothed image of input image(binary in nature)
 */

Mat vertical_smoothing(Mat image,int whitespace)
{
	int i,j,k;
	
	Mat vsmoothedimage;
	image.copyTo(vsmoothedimage);
	
	for(i=0;i<image.rows;i++)
	{
		for(j=0;j<image.cols;j++)
		{			
			if(image.data[(i*image.cols)+j]==0)
			{
				for(k=i+1;k<(i+1+whitespace);k++)
				{
					if(k<vsmoothedimage.rows)
					{					
						vsmoothedimage.data[(k*vsmoothedimage.cols)+j]=0;
					}
					else 
						break;
					
				}
			}
		}
	}
	
	return (vsmoothedimage);	
}


/**
 * @function vertical_gapfilling
 * @param input an image(binary) in Mat format and integer value whitespace that need to be gap filled or filled up
 * @brief with this function gap btween two foreground pixel is filled only if the gap between two foreground pixel in vertival direction has a gap less than or equal to whitespace
 * @brief it produce gap filled image of the input image in vertical direction by filling up with foreground with whitespace amount
 * @return Return vertically gap-filled image of input image(binary in nature)
 */


Mat vertical_gapfilling(Mat image,int whitespace)
{
	int i,j,k,l;
	
	Mat vgapfilled;
	image.copyTo(vgapfilled);
	for(i=0;i<image.rows;i++)
	{
	  for(j=0;j<image.cols;j++)
	  {
	    if(image.data[(i*image.cols)+j]==0)
	    {
	      if( (i+whitespace)<vgapfilled.rows)
	      {
		for(k=i+whitespace;k>=i+1;k--)
		{
		  if(image.data[(k*image.cols)+j]==0)
		  {
		    for(l=k;l>=i+1;l--)
		      vgapfilled.data[(l*vgapfilled.cols)+j]=0;
		    break;
		  }
		}
	      }
	      else
		break;
	    }
	  }
	}
	
	return (vgapfilled);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/*-----------------------------Statistical function--------------------------------*/


/**
 * @function FindMean
 * @param input Single Channel Mat data
 * @brief Calculate Mean of Given data array
 * @return mean(double) of the given array
 */



double FindMean(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double mean;
  double sum;
  sum = 0.0;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data_size;i++)
    sum = sum + (data.at<double>(0,i));
  mean = sum/data_size;
  
  return(mean);
}


/**
 * @function FindVar
 * @param input Single Channel Mat data
 * @brief Calculate variance of Given data array
 * @return Variance(double) of the given array
 */


double FindVar(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double var,mean;
  mean = FindMean(data);
  double temp;
  double sum=0.0;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data_size;i++)
  {
    temp = data.at<double>(0,i) - mean;
    sum = sum + (temp * temp);
  }
  var = sum/data_size;
  
  return(var);
}


/**
 * @function FindStdDev
 * @param input Single Channel Mat data
 * @brief Calculate Standard Deviation of Given data array
 * @return Standard Deviation(double) of the given array
 */


double FindStdDev(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double std_dev,var;
  var = FindVar(data);
  std_dev = sqrt(var);
  
  return(std_dev);
}


/**
 * @function FindSkew
 * @param input Single Channel Mat data
 * @brief Calculate Skewness of Given data array
 * @return Skewness(double) of the given array
 */


double FindSkew(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double skew;
  double sum = 0.0;
  double mean;
  double temp;
  double std_dev;
  int data_size;
  data_size = data.rows*data.cols;
  mean = FindMean(data);
  
  for(int i=0;i<data_size;i++)
  {
    temp = data.at<double>(0,i) - mean;
    sum = sum + (temp * temp * temp);
  }
  sum = sum / data_size;
  std_dev = FindStdDev(data);
  skew = sum/(std_dev * std_dev *std_dev);
  
}


/**
 * @function FindMinElementPosi
 * @param input Single Channel Mat data and pointer to min element  and its position
 * @brief Calculate Min value of Given data array and its position
 * 
 */



void FindMinElementPosi(Mat data, double *value, int *posi)
{
  data.convertTo(data,CV_64FC1);
  double min_element;
  min_element = data.at<double>(0,0);
  int min_posi;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data.rows;i++)
  {
    for(int j=0;j<data.cols;j++)
    {
      if(data.at<double>(i,j)<=min_element)
      {
	min_element = data.at<double>(i,j);
	min_posi = i*data.cols+j;
      }
    }
  }
  
  *value = min_element;
  *posi = min_posi;
  
}


/**
 * @function FindMaxElement
 * @param input Single Channel Mat data and pointer to max element and pointer to position
 * @brief Calculate Max value of Given data array and its position
 * 
 */



void FindMaxElementPosi(Mat data, double *value, int *posi)
{
  data.convertTo(data,CV_64FC1);
  double max_element;
  max_element = data.at<double>(0,0);
  int max_posi = 0;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data.rows;i++)
  {
    for(int j=0;j<data.cols;j++)
    {
      if(data.at<double>(i,j)>=max_element)
      {
	max_element = data.at<double>(i,j);
	max_posi = i*data.cols+j;
      }
    }
  }
  
  *value = max_element;
  *posi = max_posi;
  
}


/**
 * @function FindHistogram
 * @param input Single Channel Mat data
 * @brief Calculate Histogram of the data
 * @return Histogram data of the element
 */



Mat FindHistogram(Mat data)
{
  
 
  Mat HistData;
  double max_elem;
  int max_posi;
 
  FindMaxElementPosi(data,&max_elem,&max_posi);
  
  bool uniform = true; bool accumulate = false;
  int histSize = (int)max_elem;
  printf("HistSize is %d\t%lf\n",histSize,max_elem);
 // int histSize = 256;
  /// Set the ranges
  float range[] = { 0, histSize } ;
  const float* histRange = { range };
  
  Mat ConvertedData;
  data.convertTo(ConvertedData,CV_8UC1);
  
  /// Compute the histograms:
  calcHist( &ConvertedData, 1, 0, Mat(), HistData, 1, &histSize, &histRange, uniform, accumulate );
  
 
  return(HistData);
  
}

/**
 * @function DrawHistogram
 * @param input Single Channel Mat data
 * @brief Calculate Histogram of the data and Draw it
 * 
 */



void DrawHistogram(Mat data)
{
  Mat Histogram,NormalizedHistogram;
  
  Histogram = FindHistogram(data);
  
  double max_elem;
  int max_posi;
 
  FindMaxElementPosi(data,&max_elem,&max_posi);
  
  int histSize = (int)max_elem;
  //int histSize = 256;
  
  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
  
  //Histogram.convertTo(Histogram,CV_8UC1);

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(Histogram, Histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
 
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(Histogram.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(Histogram.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
  }
 
 
  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );

  waitKey(0);

  
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/**
 * @function FindImageInverse
 * @param input Mat data(image)
 * @brief Calculate inverse of a given image (255 - image.data[i])
 * @return inverse image(Mat)
 */

Mat FindImageInverse(Mat image)
{
  Mat InverseImage;
  image.copyTo(InverseImage);
  for(int i=0;i<image.rows*image.cols;i++)
  {
    for(int j=0;j<image.channels();j++)
    {
      InverseImage.data[i*image.channels()+j] = 255 - image.data[i*image.channels()+j];
    }
  }
  return(InverseImage);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/



BeanStucture * CreateBean(int NumberOfBean, int MaxElement)
{
  BeanStucture *Beans;
  Beans = (BeanStucture *)malloc(NumberOfBean*sizeof(BeanStucture));
  
  int Middle;
  int k;
  k =(int) MaxElement/NumberOfBean;
  
  for(int i=0;i<NumberOfBean;i++)
  {
    Beans[i].BeanNumber = i;
    Beans[i].MaxElement = MaxElement;
    Beans[i].middle = i*k;
    Beans[i].UpperBound =(int) (Beans[i].middle + (k/2));
    Beans[i].UpperBound = Beans[i].UpperBound%MaxElement;
    Beans[i].LowerBound = (int) (Beans[i].middle - (k/2));
    Beans[i].LowerBound = MaxElement + Beans[i].LowerBound;
    Beans[i].LowerBound = Beans[i].LowerBound%MaxElement;
  }
  
  return(Beans);
}


int FindOpositeBean(int BeanNumber, int NumberOfBean)
{
  int OpositeBean;
  OpositeBean =(int) BeanNumber + NumberOfBean/2;
  OpositeBean = OpositeBean%NumberOfBean;
}

int FindBeanNumber(int angle, int MaxElement, int NumberOfBean)
{
  //int MaxElement = Beans[0].MaxElement;
  int BeanedAngle = MaxElement/NumberOfBean;
  int temp_angle =(int) angle + (BeanedAngle/2);
  int BeanNum;
  BeanNum =(int) (temp_angle/BeanedAngle);
  BeanNum = BeanNum % NumberOfBean;
  
  return(BeanNum);
}



Point2i FindNextPixel8Bean(Point2i p, int Bean)
{
  Point2i next;
  if(Bean == 0)
  {
    next.y = p.y;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 1)
  {
    next.y = p.y - 1;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 2)
  {
    next.y = p.y - 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 3)
  {
    next.y = p.y - 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 4)
  {
    next.y = p.y;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 5)
  {
    next.y = p.y + 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 6)
  {
    next.y = p.y + 1;
    next.x = p.x;
    return(next);
  }
  else
  {
    next.y = p.y + 1;
    next.x = p.x + 1;
    return(next);
  }
}


Point2i FindNextPixel12Bean(Point2i p, int Bean)
{
  Point2i next;
  if(Bean == 0)
  {
    next.y = p.y;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 1)
  {
    next.y = p.y - 1;
    next.x = p.x + 2;
    return(next);
  }
  else if(Bean == 2)
  {
    next.y = p.y - 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 3)
  {
    next.y = p.y - 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 4)
  {
    next.y = p.y - 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 5)
  {
    next.y = p.y - 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 6)
  {
    next.y = p.y;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 7)
  {
    next.y = p.y + 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 8)
  {
    next.y = p.y + 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 9)
  {
    next.y = p.y + 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 10)
  {
    next.y = p.y + 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 11)
  {
    next.y = p.y + 1;
    next.x = p.x + 2;
    return(next);
  }
}


Point2i FindNextPixel16Bean(Point2i p, int Bean)
{
  Point2i next;
  if(Bean == 0)
  {
    next.y = p.y;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 1)
  {
    next.y = p.y - 1;
    next.x = p.x + 2;
    return(next);
  }
  else if(Bean == 2)
  {
    next.y = p.y - 1;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 3)
  {
    next.y = p.y - 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 4)
  {
    next.y = p.y - 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 5)
  {
    next.y = p.y - 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 6)
  {
    next.y = p.y - 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 7)
  {
    next.y = p.y - 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 8)
  {
    next.y = p.y;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 9)
  {
    next.y = p.y + 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 10)
  {
    next.y = p.y + 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 11)
  {
    next.y = p.y + 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 12)
  {
    next.y = p.y + 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 13)
  {
    next.y = p.y + 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 14)
  {
    next.y = p.y + 1;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 15)
  {
    next.y = p.y + 1;
    next.x = p.x + 2;
    return(next);
  }
}


Ray FindStrokeWidth(Point2i p,int Bean, Mat BoundaryImage, Mat GradBean, Mat GrayImage, int NumberOfBean)
{
 // printf("In FindStroke with y = %d and x = %d and row = %d and col = %d and bean = %d\n",p.y,p.x,BoundaryImage.rows,BoundaryImage.cols,Bean);
  Ray TempRay;
  TempRay.p = p;
  TempRay.q = p;
  TempRay.dist = 0;
  Point2i Next;
  Next = p;
  int OpositeBean = FindOpositeBean(Bean, NumberOfBean);
  int bean1,bean2;
  int m,n;
  while(1)
  {
    //Next = FindNextPixel8Bean(Next,Bean);
    
    if(NumberOfBean == 8)
      Next = FindNextPixel8Bean(Next,Bean);
    else if(NumberOfBean == 12)
      Next = FindNextPixel12Bean(Next,Bean);
    else if(NumberOfBean == 16)
      Next = FindNextPixel16Bean(Next,Bean);
    
    TempRay.points.push_back(Next);
    TempRay.dist = TempRay.dist + 1;
   // printf("Next y = %d and x = %d and row = %d and col = %d\n",Next.y,Next.x,BoundaryImage.rows,BoundaryImage.cols);
    
    if(validate(Next.y,Next.x,BoundaryImage.rows,BoundaryImage.cols))
    {
     // printf("hello\n");
      if(GrayImage.at<uchar>(Next.y,Next.x) == 255)
      {
	//printf("hello1\n");
	//TempRay.dist = 0;
	//break;
	
	for( m=Next.y-1;m<=Next.y+1;m++)
	{
	  for( n=Next.x-1;n<=Next.x+1;n++)
	  {
	    if(validate(m,n,BoundaryImage.rows,BoundaryImage.cols))
	    {
	      if(BoundaryImage.at<uchar>(m,n) == 0)
	      {
		Next.y = m;
		Next.x = n;
		break;
	      }
	    }
	  }
	  if(n<Next.x+2)
	   break;
	}
	if(m==Next.y+2 && n==Next.x+2)
	{
	//  printf("helloo121\n");
	  TempRay.dist = 0;
	  break;
	}
	//printf("hello3neighbour\n");
      }
      
      
      
      if(BoundaryImage.at<uchar>(Next.y,Next.x) == 0)
      {
	//printf("hello12\n");
	bean1 = (OpositeBean + 1)%NumberOfBean;
	bean2 = (OpositeBean + NumberOfBean-1)%NumberOfBean;
	if(GradBean.at<int8_t>(Next.y,Next.x) == OpositeBean || GradBean.at<int8_t>(Next.y,Next.x) == bean1 || GradBean.at<int8_t>(Next.y,Next.x) == bean2)
	{
	//  printf("helloq\n");
	  TempRay.q = Next;
	  TempRay.CalcEcluiDist();
	 // printf("hello1\n");
	  break;
	}
	else
	{
	 // printf("hello7\n");
	 // TempRay.q = Next;
	 // TempRay.CalcEcluiDist();
	  TempRay.dist = 0;
	  break;
	}
      }
      
      if(TempRay.dist > (GrayImage.cols/3) || TempRay.dist > (GrayImage.rows/3))
      {
	//printf("hello123\n");
	//TempRay.dist = 0;
	break;
      }
      
      
    }
    else
    {
     // printf("hello1234\n");
      TempRay.dist = 0;
      break;
    }
  }
  
 // printf("Done and dist is %d\n",TempRay.dist);
  return(TempRay);
}



/*-------------------------------------------------------------------------------------------------------------------------------------------*/



/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./swt <image-location-with name>" << std::endl; }


/*-------------------------------------------------MAIN--------------------------------------------------------------------------------------*/


 
int main(int argc, char *argv[])
{
  if( argc != 2 )
  { readme(); 
    return -1; 
  }
  
  src = imread(argv[1],1);
  int row,col;
  row = src.rows;
  col = src.cols;
  
  char *substring,*name;
  
  substring = input_image_name_cut(argv[1]);
  makedir(substring);
  
   int i,j; 
   int bflag = 1;
   printf("dark on bright press 1 \n bright on dark press 2\n");
   
   scanf("%d",&bflag);
   if(bflag == 2)
     src = FindImageInverse(src);
   
  Mat ForeGroundImage;
  ForeGroundImage = foreground_masked_image(src);
  
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"ForeGroundImage.png");
  imwrite(name,ForeGroundImage);
   
  Mat BinaryImage;
  BinaryImage = binarization(src,2);
  
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"BinaryImage.png");
  imwrite(name,BinaryImage);
  
  Mat GrayImage;
  cvtColor(ForeGroundImage,GrayImage,CV_BGR2GRAY);
  
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"GrayImage.png");
  imwrite(name,GrayImage);
  
  /*
  Mat detected_edges;
  blur( GrayImage, detected_edges, Size(3,3) );
 
  //GaussianBlur( GrayImage, detected_edges, Size(3,3), 0 );
  
  int lowThreshold;
  int ratio = 3;
  int kernel_size = 3;
  
  
  
 lowThreshold = 10;
  /// Canny detector

  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
  
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"CannyImage.png");
  imwrite(name,detected_edges);
 
 */
 
  Mat BoundaryImage;
  BoundaryImage = boundaryextraction(BinaryImage);
  
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"BoundaryImage.png");
  imwrite(name,BoundaryImage);
  
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;
  
   /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( GrayImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  //convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( GrayImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
 // convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
 // addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  int no_edge_pixel = 0;
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.data[i*col+j]==0)
      {
	no_edge_pixel = no_edge_pixel + 1;
      }
    }
  }
  
  Mat grad = Mat::zeros(GrayImage.rows,GrayImage.cols,CV_64FC1);
  Mat TempGradxyGradDirMag;
  TempGradxyGradDirMag = Mat(GrayImage.rows,GrayImage.cols,CV_64FC4);
  Mat NormalizeGradxy;
  NormalizeGradxy = Mat(GrayImage.rows,GrayImage.cols,CV_64FC2);
  int k = 0;
  double x_dir,y_dir;
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	x_dir = grad_x.at<double>(i,j);
	//TempGradx.at<double>(i,j) = x_dir;
	y_dir = grad_y.at<double>(i,j);
	//TempGrady.at<double>(i,j) = y_dir;
	grad.at<double>(i,j) = (atan2(y_dir, x_dir)*180)/PI;
	if(grad.at<double>(i,j) < 0)
	  grad.at<double>(i,j) = 180 - grad.at<double>(i,j);
	TempGradxyGradDirMag.at<Vec4d>(i,j)[0] = x_dir;
	TempGradxyGradDirMag.at<Vec4d>(i,j)[1] = y_dir;
	TempGradxyGradDirMag.at<Vec4d>(i,j)[2] = (atan2(y_dir, x_dir)*180)/PI;
	TempGradxyGradDirMag.at<Vec4d>(i,j)[3] = sqrt((x_dir*x_dir)+(y_dir*y_dir));
	NormalizeGradxy.at<Vec2d>(i,j)[0] = x_dir/TempGradxyGradDirMag.at<Vec4d>(i,j)[3];
	NormalizeGradxy.at<Vec2d>(i,j)[1] = y_dir/TempGradxyGradDirMag.at<Vec4d>(i,j)[3];
	
	//printf("grad x = %lf\t grad y = %lf and \tgrad dir = %lf\n",x_dir,y_dir,grad.at<double>(i,j));
      }
    }
  }
  
  double max_elem,min_elem;
  int posi;
  
  FindMaxElementPosi(grad,&max_elem,&posi);
  FindMinElementPosi(grad,&min_elem,&posi);
  
  printf("MAX Angle is %lf\tand MIN Angle is %lf\n",max_elem,min_elem);
 
   
  Mat absgrad = Mat(grad.rows,grad.cols,CV_16UC1);
  
  
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	absgrad.at<int16_t>(i,j) =(int16_t) floor(grad.at<double>(i,j));
      }
    }
  }
  Mat GradHist = Mat::zeros(360,1,CV_32FC1);
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	GradHist.at<float>(absgrad.at<int16_t>(i,j),0) = GradHist.at<float>(absgrad.at<int16_t>(i,j),0) + 1;
      }
    }
  }
  
  FILE *fp;
   name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"gradhist.xls");
   fp = fopen(name,"w");
  
  for(int i=0;i<360;i++)
  {
    GradHist.at<float>(i,0) = GradHist.at<float>(i,0) / no_edge_pixel;
    fprintf(fp,"\n\t%f",GradHist.at<float>(i,0));
  }
  fclose(fp);

//    namedWindow( "grad", CV_WINDOW_KEEPRATIO );
//    imshow("grad", grad);
//    waitKey(0);
  
  Mat abs_grad;
  convertScaleAbs( grad, abs_grad );
  
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"edge_gradimage.png");
  imwrite(name,abs_grad);
  
 
 printf("Give Number of Bean:\n");
 printf("Press 1 For 8 Bean\nPress 2 For 12 Bean\nPress 3 For 16 Bean\n");
 int flag;
 scanf("%d",&flag);
  

  int BeanNum;
  int NumberOfBean;
  
  
  
  if(flag == 1)
    NumberOfBean = 8;
  else if(flag == 2)
    NumberOfBean = 12;
  else if(flag == 3)
    NumberOfBean = 16;
  else
  {
    printf("Bean not Selected Properly\n Choosing Default Bean = 8\n");
    NumberOfBean = 8;
  }
  
  
  BeanStucture *BeanData;
 
 BeanData = CreateBean(NumberOfBean,360);
  
  
  Mat GradBean = Mat(grad.rows,grad.cols,CV_8UC1);
 // Mat GradBeanHist = Mat::zeros(NumberOfBean,1,CV_32FC1);
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	  BeanNum = FindBeanNumber(absgrad.at<int16_t>(i,j),360,NumberOfBean);
	  
	  GradBean.at<int8_t>(i,j) = BeanNum;
	  printf("Bean number for %d is %d\n",absgrad.at<int16_t>(i,j),GradBean.at<int8_t>(i,j));
	//  GradBeanHist.at<float>(BeanNum,0) = GradBeanHist.at<float>(BeanNum,0) + 1;
      }
    }
  }
  /*
  name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
  name = CreateNameIntoFolder(substring,"gradbeanhist.xls");
   fp = fopen(name,"w");
  
  for(int i=0;i<8;i++)
  {
    GradBeanHist.at<float>(i,0) = GradBeanHist.at<float>(i,0) / no_edge_pixel;
    fprintf(fp,"\n\t%f",GradBeanHist.at<float>(i,0));
  }
  fclose(fp);
  */
  Ray tempRay;
  int NumberOfStrokes = 0;
  vector<Ray> strokes;
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	 int bean;
	 bean = GradBean.at<int8_t>(i,j);
	 Point2i p;
	 p.x = j;
	 p.y = i;
	// printf("Bean Number = %d\n",GradBean.at<int8_t>(i,j));
	// printf("Bean Number = %d\n",bean);
	 tempRay = FindStrokeWidth(p,bean,BoundaryImage,GradBean,GrayImage, NumberOfBean);
	// printf("Dist = %d\n",tempRay.dist);
	// printf("Done upto hear\n");
	 if(tempRay.dist > 0)
	 {
	  strokes.push_back(tempRay);
	  NumberOfStrokes = NumberOfStrokes + 1;
	 }
      }
    }
  }
  
  if(NumberOfStrokes > 1)
  {
    
    Mat StrokeWidth = Mat(NumberOfStrokes,1,CV_16UC1);
    
    Mat drawing = Mat::zeros(src.rows,src.cols,CV_8UC3);
  
    i = 0;
    int c1,c2,c3;
    for(vector<Ray>::iterator pid=strokes.begin();pid!=strokes.end();pid++)
    {
      if(pid->dist > 2)
      {
	StrokeWidth.at<int16_t>(i,0) = pid->dist;
	i = i + 1;
	//Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	//printf("For Point (%d %d) Stroke Width is %d\n",pid->p.y,pid->p.x,pid->dist);
	c1 = rng.uniform(0, 255);
	c2 = rng.uniform(0, 255);
	c3 = rng.uniform(0, 255);
	for(vector<Point2i>::iterator m=pid->points.begin();m!=pid->points.end();m++)
	{
	  drawing.at<Vec3b>(m->y,m->x)[0] = c1;
	  drawing.at<Vec3b>(m->y,m->x)[1] = c2;
	  drawing.at<Vec3b>(m->y,m->x)[2] = c3;
	}
	
      }
    }
    name = (char *) malloc ( 2001 * sizeof(char));
	if(name == NULL)
	{
	  printf("Memory can not be allocated\n");
	  exit(0);
	}
    name = CreateNameIntoFolder(substring,"stroke.png");
    imwrite(name,drawing);
    
    
    name = (char *) malloc ( 2001 * sizeof(char));
	if(name == NULL)
	{
	  printf("Memory can not be allocated\n");
	  exit(0);
	}
    name = CreateNameIntoFolder(substring,"StrokeHist.xls");
    fp = fopen(name,"w");
    
    Mat SWHist = FindHistogram(StrokeWidth);
    for(int i=0;i<SWHist.rows;i++)
    {
      for(int j=0;j<SWHist.cols;j++)
      {
	SWHist.at<float>(i,j) = SWHist.at<float>(i,j)/NumberOfStrokes;
	fprintf(fp,"\n\t%f",SWHist.at<float>(i,j));
      }
    }
    fclose(fp);
    
    //DrawHistogram(StrokeWidth);
    //waitKey(0);
  
  }
  /*
   namedWindow( "grad", CV_WINDOW_KEEPRATIO );
    imshow("grad", src);
  
  namedWindow( "stroke", CV_WINDOW_KEEPRATIO );
    imshow("stroke", drawing);
    waitKey(0);
  */  

  return 0;
}