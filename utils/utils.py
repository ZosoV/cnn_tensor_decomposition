from ctypes import *
import numpy as np
import cv2
from utils.structures import *
from utils.transformation import *
import tensorflow as tf
import matplotlib.pyplot as plt

def wrap_function(libpath, funcname, restype, argtypes):   
    """Simplify wrapping ctypes functions"""
    print("[INFO] Loading c++ library: ",libpath)
    lib = cdll.LoadLibrary(libpath)   # load the lib of a specific path
    func = lib.__getattr__(funcname)  # get the function git the specific name
    func.restype = restype            # define the type of results
    func.argtypes = argtypes          # define the type of the arguments
    return func

def print_numpy(np_arr):
    h, w = np_arr.shape
    for i in range(h):
        for j in range(w):
            print(np_arr[i,j], end=" ")
        print("")
    print("-------") 

def print_numpy3D(np_arr):
  h, w, c = np_arr.shape
  for k in range(c):
    for i in range(h):
      for j in range(w):
        print(np_arr[i,j,k], end=" ")
      print("")
    print("-------") 

def store_images(list_images):
    #Save the kernels in a folder  
    for idx, img in enumerate(list_images):
        print("Saving Img Kernel: {} Min: {} Max: {}".format(idx,np.amin(img), np.amax(img)))
        cv2.imwrite("py_results/kernel_{}.jpg".format(idx),img) 

def pixel_scale(input, method_str = "normalize"):
    result = input
    if (method_str == "normalize"):
        result = result / 255.0
    elif (method_str == "center"):
        m = result.mean() if isinstance(result, np.ndarray) else result.numpy().mean()
        result = result - m
    elif (method_str == "standard"):
        m = result.mean() if isinstance(result, np.ndarray) else result.numpy().mean()
        s = result.std() if isinstance(result, np.ndarray) else result.numpy().std()
        result = (result - m) / s

    return result

def print_weights(model, type_layer):
    if (type_layer == 'conv'):
        print(f'Cov weights - First Head')
        for layer in model.layers:
            if 'conv' in layer.name: 
                print(layer.get_weights()[0][:,:,0,0])
                print("-----------")
    elif (type_layer == 'dense'):
        print(f'Dense weights - chunck 4x4')    
        for layer in model.layers:
            if 'dense' in layer.name:
                print(layer.get_weights()[0][:4,:4])
                print("-----------")


def custom_mse(x,y, axis = (0,1,2), event_axis = None):

  if not (event_axis is None):
    event_num = x.shape[event_axis]

    result = np.zeros(shape = (event_num,event_num))
    for i in range(event_num):
      for j in range(event_num):
        result[i,j] = tf.math.reduce_mean(tf.square(x[:,:,:,i] - y[:,:,:,j]), axis=axis)
    
    result = tf.convert_to_tensor(result)
  else:

    result = tf.math.reduce_mean(tf.square(x - y), axis=axis)

  return result