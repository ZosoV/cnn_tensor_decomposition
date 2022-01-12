#include <eigen3/Eigen/Dense>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"

#include <string.h>
#include <iostream>
#include <math.h> 

using namespace std;
using namespace cv;
using namespace Eigen;         

struct image_float_t{
    int H;
    int W;
    int C;
    float *data;
};

struct cov_matrix{
    int H;
    int W;
    int C;
    int r1;
    int r2;
    int r3;
    int kernel_size;
    int stride;
    double *data;
};

struct kernel_tensor {
    int size;
    int channel;
    double * data;  
};

struct kernel_block {
    int num;
    kernel_tensor * kernels;
};

void print_kernel(kernel_tensor kernel, int N);

float get_image_pixel_hwc(const image_float_t& im, int h, int w, int c);

Mat image_float_to_mat(const image_float_t& im);

image_float_t mat_to_image_float(const Mat& mat );

kernel_tensor mats2kernel( vector<Mat> mats, int cnt_kn);

Mat image_char_to_mat(const kernel_tensor &im);

Mat image_float2mat(const image_float_t& im);

MatrixXd createXMatrix(const image_float_t& im);

MatrixXd rebuildSMatrix(const cov_matrix& input);

kernel_tensor matrix2kernel( MatrixXd tensorspace, int kn_size, int ch);