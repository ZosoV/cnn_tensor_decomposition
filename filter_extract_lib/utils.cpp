#include "cov6d.h"

float get_image_pixel_hwc(const image_float_t& im, int h, int w, int c){
    //(1) bgr, hwc, 0-255    h*W*C+w*C+c     int index = h*im.W*im.C + (w*im.C + c); 
    int index = h*im.W*im.C + (w*im.C + c); 
    return im.data[index];
}

Mat image_float_to_mat(const image_float_t& im){
    // bgr,hwc, 0-255
    Mat mat(im.H, im.W, CV_8UC3);
    for(int h=0; h< im.H; ++h)
    {
        Vec3b *p = mat.ptr<Vec3b>(h);
        for(int w=0; w < im.W; ++w)
        {
            for(int c=0; c< im.C; ++c)
            {
                // b,g,r
                (*p)[c] = (unsigned char)get_image_pixel_hwc(im, h, w, c); 
            }
            p++;
        }
    }
    return mat;

}

Mat kernel_to_mat(const kernel_tensor &im){
    // bgr,hwc, 0-255
    Mat mat(im.size, im.size, CV_8UC3, im.data);
    return mat; // data in python
}

void print_kernel(kernel_tensor kernel, int N){
    cout << "Values of c+++" << endl;
    for (int i = 0; i <  N; i++){
        cout << kernel.data[i] << " ";
    }
    cout << endl;
}

kernel_tensor mats2kernel( vector<Mat> mats, int cnt_kn){
    
    //Create the data of the kernel
    double* data = new double[mats[0].rows * mats[0].cols * mats.size()];

    //Iterate over the mats, and flatten the pixels values in a data array
    int count = 0;
    for (int i = 0; i < mats[0].rows; i++){
        for (int j = 0; j < mats[0].cols; j++){
            for (int k = 0; k < mats.size(); k++){
                data[count] = (double) mats[k].at<uchar>(i,j);
                count += 1;
            }   
        } 
    }

    //Take the size and channels of the kernel
    int size_data = mats[0].rows;
    int ch = mats.size();
    
    //Create a kernel tensor and return it
    kernel_tensor kn = { size_data, ch, data };


    return kn;
}




Mat image_float2mat(const image_float_t& im){
    // Create stacked vector of cv::Mats
    vector<Mat> stacked_mats;
    stacked_mats.reserve(im.C);  // Reserve space for efficiency

    // Stack the im.C
    for (int k = 0; k < im.C; ++k) {
        
        //Create temporal pixels to stores the values of one channel
        float* pixels = new float[im.H * im.W];
        int count = 0;

        //Iterate over a channel
        for (int i = 0; i < im.H; ++i) {
            for (int j = 0; j < im.W; ++j) {

                //Take the pixel value an store in the float*
                float pixel_value = (float) im.data[i*im.W*im.C + (j*im.C + k)];
                pixels[count] = pixel_value;
                count++;
            }
        }

        //Create a channel mat using the temporal float *        
        Mat channel = Mat(im.H, im.W, CV_32FC1, pixels).clone();
        // cout << "M"<< k <<" = " << endl << " "  << channel << endl << endl;

        //stack the channel
        stacked_mats.push_back(channel);
    }

    // Stores output matrix
    Mat output;

    // Create the output matrix merging all the channels
    merge(stacked_mats, output);

    return output;
}

MatrixXd createXMatrix(const image_float_t& im){

    // cout << "channels: " << im.C << " " << im.W*im.C << endl;

    MatrixXd X(im.H, im.W*im.C);
    
    // cout << "rows: " << X.rows() << " cols: " << X.cols() << endl;
    for (int k = 0; k < im.C; k++){
        double* tmp = new double[im.H*im.W];
        
        int count = 0;
        for (int i = 0; i < im.H; i++){
            for (int j = 0; j < im.W; j++){
                tmp[count] = im.data[i*im.W*im.C + (j*im.C + k)];

                count++;
            }
        }

        MatrixXd m_tmp = Map<MatrixXd>( tmp, im.H, im.W );
        // cout << "m_tmp: " <<" = " << endl << " "  << m_tmp << endl << endl;

        // cout << "X in func: " <<" = " << " "  << k*im.W << " "<< " " <<  im.H << " " << im.W << endl;  
        X.block(0,k*im.W,im.H,im.W) = m_tmp.transpose();
        
    }

    return X;
}

MatrixXd rebuildSMatrix(const cov_matrix& input){

    int S_size = input.kernel_size * input.kernel_size * input.C;

    MatrixXd S = Map<MatrixXd>( input.data, S_size, S_size );

    // cout << "S in C++: " <<" = " << endl << " "  << S << endl << endl;

    return S;
}


kernel_tensor matrix2kernel( MatrixXd tensorspace, int kn_size, int ch){
    
    //Create the data of the kernel
    double* data = new double[kn_size * kn_size * ch];

    //Iterate over the mats, and flatten the pixels values in a data array
    int count = 0;
    for (int i = 0; i < kn_size; i++){
        for (int j = 0; j < kn_size; j++){
            for (int k = 0; k < ch; k++){
                data[count] = (double) tensorspace(i,j + k*kn_size);
                count += 1;
            }   
        } 
    }
    
    //Create a kernel tensor and return it
    kernel_tensor kn = { kn_size, ch, data };


    return kn;
}