# import math 

# soft_max = [0.7, 0.2, 0.1]
# target = [1,0,0]

# loss = - (math.log(soft_max[0]) * target[0] +
#         math.log(soft_max[1]) * target[1] +
#         math.log(soft_max[2]) * target[2]) 

# # print(loss) 

# # the same as doing only 
# loss = -(math.log(soft_max[0]) * target[0])
# # print(loss)

# # 0.35667494393873245 is the loss of 0.7 

import numpy as np

softmax_outputs = np.array([[0.7, 0.2, 0.1], 
                            [0.1, 1.5, 0.2],
                            [0.2, 0.9, 0.5]])

class_target = [0,1,1]

print(-np.log(softmax_outputs
              [range(len(softmax_outputs)), class_target]))
