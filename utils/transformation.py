import numpy as np
from utils.structures import *

def numpy2image_float(bgr):
    # get the shape of the numpy image
    (h,w,c) = bgr.shape
    print("Shape numpy2image_float: {} {} {}".format(h,w,c))

    # flatten an image to a continuous array
    arr = np.ascontiguousarray(bgr.flat, dtype=np.float32)

    # convert to the POINTER(c_float) data type accepted by c++
    data = arr.ctypes.data_as(POINTER(c_float))

    # create the IMAGE FLOAT structure and return it
    im = IMAGE_FLOAT(h, w, c, data) # bgr,hwc,0-255
    return im, arr #  return `arr` to avoid python freeing memory    

def kernel2numpy(num_k, kernel):
    #Transforming the kernel to a numpry array
    image = np.ctypeslib.as_array(kernel.data, shape=(kernel.size,kernel.size,kernel.channels))
    #image = image / 255.0
   
    return image

def knblock2numpys(kn_block):
    print("[INFO] Total number of generated kernels: {}".format(kn_block.num))
    kns_images = []
    for i in range(kn_block.num):
        kernel = kn_block.kernels[i]
        kn_img = kernel2numpy(i,kernel)
        kns_images.append(kn_img)

    return kns_images

def numpy2cov_variance(S, tensor_shape, kernel_size, stride, num_r ):
    # get the shape of the numpy image
    S_size = S.shape[0]
    
    r1, r2, r3 = num_r
    (_, h_input, w_input, c_input) = tensor_shape
    
    # flatten an image to a continuous array
    arr = np.ascontiguousarray(S.flat, dtype=np.double)

    # convert to the POINTER(c_float) data type accepted by c++
    data = arr.ctypes.data_as(POINTER(c_double))

    # create the IMAGE FLOAT structure and return it
    cov_matrix = COV_MATRIX(h_input, w_input, c_input, r1, r2, r3, kernel_size, stride, data) # S,hwc,0-255
    return cov_matrix, arr #  return `arr` to avoid python freeing memory
