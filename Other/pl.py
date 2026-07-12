import numpy as np  # import numpy so we can do math easily
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()  # initializes the random number generator so we get the same random numbers every time we run the code

#Dense layer
class Layer_Dense: #A dense layer means every input connects to every neuron.
    def __init__(self,n_inputs,n_neurons): # When I first build this machine, what should I put inside it?
        self.weights = 0.10 * np.random.rand(n_inputs,n_neurons) #creating random weight# 4 inputs, 3 neurons
        self.biases = np.zeros((1,n_neurons)) # 1 bias per neuron, so 3 biases
        
    def forward(self, inputs): # When I give the machine something, what should it do with it?
        self.output = np.dot(inputs, self.weights) + self.biases
    
    #input × weights + bias = output score
  
  
# ReLU activtion removes negative values and keeps positive values the same, 
class Activation_ReLU:  
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # ReLU activation function,  if i is less than 0, 0 is the max

# -5 becomes 0
# 3 stays 3

#SoftMax 
class Activation_Softmax:
    def forward(self, inputs): # Softmax receives the final raw scores.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Subtract the max value from each input to prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalize the values to get probabilities
        self.output = probabilities

#raw scores: [4.8, 1.21, 2.385]

#softmax: [0.895, 0.025, 0.080]

class Loss:
    def calculate(self, output, y):  # small calculator for computing loss, the output and y we feed the machine
        sample_losses = self.forward(output, y)  # output and y is what we feed the machine
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss): # feeding it loss infomration 
    def forward(self, y_pred, y_true): #predicted values and true values
        samples = len(y_pred) # counts how many prodections in the smample in this case 100
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip the predicted values to prevent log(0)
        # 1e-7 is a tiny number (0.0000001)
        # 1 - 1e-7 is almost 1 (0.9999999)
        

        if len(y_true.shape) == 1: # If 
            correct_confidences = y_pred_clipped[range(samples), y_true]
            
            
        elif len(y_true.shape) == 2: # If the labels are one-hot encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
class Predictions:
    def calculator (self, output, y):
        P_Calculator = np.argmax(output, axis=1)
        accuracy = np.mean(P_Calculator == y)
        return accuracy



X,y= spiral_data(samples=100, classes=3) # 100 samples per class, 3 classes

dense1 = Layer_Dense(2,3) # 2 inputs, 3 neurons
activation1 = Activation_ReLU() # ReLU activation function 

dense2 = Layer_Dense(3,3) # 3 inputs, 3 neurons
activation2 = Activation_Softmax() # Softmax activation function

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("loss:",loss)

print(activation2.output[:3])
print(y[:3])  


predictions = Predictions()
acc = predictions.calculator(activation2.output, y)
print("Accuracy:", acc)






















# X is the input data
# 3 rows = 3 examples
# 4 numbers in each row = 4 inputs
# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# input = [0,2,-1,3.3 ,-2.7, 1.1, 2.2 , -100]
# output = []

# for i in input:
#     output.append(max(0,i)) # ReLU activation function,  if i is less than 0, 0 is the max
    
    
    # if i > 0:
    #     output.append(i)
    # elif i <= 0:
    #     output.append(0)

# print(output)





































# weights = [[0.1,0.3,0.5,0.3]
#            ,[0.2,0.4,0.6,0.2]
#            ,[0.3,0.5,0.7,0.5]]

# bias  = [0.2, 0.2,0.2]

# output = np.dot(input, np.array(weights).T) + bias
# print(output) 