# import math 
# import numpy as np

# layer_outputs = [[4.8, 1.21, 2.385]]


# E = math.e 

# # Exponentiation = = make every score positive and bigger.
# exp_values = np.exp(layer_outputs) 
# # the layer ouputes are 4.8, 1.21, 2.385, so the exp values are e^4.8, e^1.21, e^2.385
# # exp_values is [121.51041752, 3.35350111, 10.85996519] to make them bigger 
 
# norm_values = exp_values / np.sum(exp_values)
# # the sum of the exp values is 135.72388382, 
# # so we divide each exp value by that to get the norm values
# # [121.51041752, 3.35350111, 10.85996519] / 135.72388382 = [0.895282, 0.024713, 0.080005]


# print(exp_values)
# print(norm_values) 


import numpy as np 
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

# print(np.sum (layer_outputs, axis=1 , keepdims=True)) # axis=1 means we sum across the rows, keepdims=True means we keep the dimensions of the original array

norm_vales = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_vales)