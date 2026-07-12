import math 

layer_outputs = [4.8, 1.21, 2.385]

E = math.e 

# Exponentiation = = make every score positive and bigger.
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
    

    
    
# Normalization  = = make every score positive and bigger. 
norm = sum(exp_values)
norm_base = []

for values in exp_values:
    norm_base.append(values/norm)
    
print(norm_base)
print(sum(norm_base)) #making sure it adds up to 1, which is what we want for probabilities.
