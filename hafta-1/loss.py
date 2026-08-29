from neuron import neuron 

def loss(prediction, target):

    loss_value = ( target - prediction) **2
    return loss_value
    
    
    
inputs = [0.5, 0.2, 0.1]
weights = [0.4, 0.3, 0.9]
bias = 0.5
target = 1.0

"""prediction = neuron(inputs, weights, bias)""loss_value = loss(prediction, target)

print("prediction:", prediction)
print("Loss:", loss_value)
"""

weight_values = []

weight = -1.0
while weight <= 1.0:
    weight_values.append(weight)
    weight += 0.05
    
loss_values = []
for weight in weight_values:
    weights = [weight,0.3,0.9]
    prediction = neuron(inputs,weights,bias)
    loss_value = loss(prediction,target)
    loss_values.append(loss_value)
    
    
    
import matplotlib.pyplot as plt 
plt.plot(weight_values,loss_values)
plt.xlabel("Weight value")
plt.ylabel("Loss value")
plt.title("Loss vs Weight")
plt.show()

    