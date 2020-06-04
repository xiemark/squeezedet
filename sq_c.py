import numpy as np

import cv2
from os import path
from os import getcwd
import ctypes


img = cv2.imread('006985.png')

img = cv2.resize(img, (1242,375), interpolation=cv2.INTER_CUBIC)
print(img.shape)

shape=(375,1242,3)
imgout =np.zeros(shape, dtype='uint8')

dir_c = getcwd()
so_lib_path= path.join(dir_c, 'run_network_dll.dll')

ModelDLL = ctypes.CDLL(so_lib_path)

input_info = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3,
                                    shape=(375,1242,3), flags='CONTIGUOUS')
output_info = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3,
                                    shape=(375,1242,3), flags='CONTIGUOUS')
ModelDLL.run_network_dll.argtypes = [input_info,output_info]
ModelDLL.run_network_dll.restype = ctypes.c_int

ModelDLL.run_network_dll(img,imgout)



cv2.imshow('image',imgout)
cv2.waitKey(6000)

