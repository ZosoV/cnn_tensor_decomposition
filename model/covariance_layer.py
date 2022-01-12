import tensorflow as tf
import os

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD

from utils.utils import *
from model.model_utils import *

#Generate the function wrapper from c++ library
get_filters = wrap_function(libpath = os.path.join(os.getcwd(), 'filter_extract_lib/libtensorfilters.so'), 
                        funcname = 'get_filters', 
                        restype = KERNEL_BLOCK, 
                        argtypes = [COV_MATRIX])


# Dictionary to control the num of kernels per layer
# according our control parameter, [remove_kernels, (r1, r2, r3)]
map_num_kernels = {
    "32init" : [4,(4,3,3)],
    "32" : [4,(3,3,4)],
    "64" : [8,(3,3,8)],
    "128" : [7,(3,3,15)]
}

#Define the covariance layer
class CovarianceLayer(tf.keras.layers.Layer):
  def __init__(self, num_filters, kernel_size, gen_stride, stride = 1, padding = 'VALID', activation=None, subtract = None):
    super(CovarianceLayer, self).__init__()
    # Init the paremeters of the covariance layer
    kernel_0, num_r = map_num_kernels[num_filters]
    self.kernel_0 = kernel_0        #the first kernel to take from the list of kernels
    self.kernel_size = kernel_size  #kernel size of all kernels
    self.gen_stride = gen_stride    #stride of the c++ function get_filters
    self.stride = stride            #stride of the convolution
    self.padding = padding          #padding of the convolution
    self.num_r = num_r              #values [r1, r2, r3] to get the filter using the function get filterrs of c++
    self.activation = activation    #activation of the convolution
    self.subtract = subtract

  def build(self, input_shape):
    num_kernels = self.num_r[0] * self.num_r[1] * self.num_r[2] - self.kernel_0 #get the number
    self.kernels_tensor = self.add_weight("kernels_tensor",
                                shape=[self.kernel_size,
                                self.kernel_size,
                                input_shape[3],
                                num_kernels],
                                dtype=np.float32)
  def call(self, input_tensor, training):
    print("[INFO] Input tensor in covariance layer {}".format(input_tensor.shape))
    print("[INFO] Generating Kernels Step ")
    
    self.set_weights([generate_kernels(get_filters, input_tensor, 
                                                self.kernel_0,
                                                self.kernel_size, 
                                                self.gen_stride,
                                                self.num_r,
                                                self.subtract)])
    print("[INFO] Ending Generating Kernels Step")

    result = tf.nn.conv2d(input_tensor,self.kernels_tensor,self.stride, self.padding)
    
    if (self.activation == 'relu'):
      result = tf.nn.relu(result)

    if (self.activation == 'leaky_relu'):
      result = tf.nn.leaky_relu(result)

    return result