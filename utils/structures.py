from ctypes import *

class IMAGE_FLOAT(Structure):
    _fields_ = [("h", c_int),
                ("w", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class KERNEL(Structure):
    _fields_ = [("size", c_int),
                ("channels", c_int),
                ("data", POINTER(c_double))]

class KERNEL_BLOCK(Structure):
    _fields_ = [("num", c_int),
                ("kernels", POINTER(KERNEL))]

class COV_MATRIX(Structure):
    _fields_ = [("h", c_int),
                ("w", c_int),
                ("c", c_int),
                ("r1", c_int),
                ("r2", c_int),
                ("r3", c_int),
                ("kernel_size", c_int),
                ("stride", c_int),
                ("data", POINTER(c_double))]