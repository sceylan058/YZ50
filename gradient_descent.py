from neuron import neuron
from loss import loss

input = [0.5]
weight = -4.0

neuron(input, [weight],0.5)
loss(neuron(input, [weight],0.5), 0.8)
 
h = 0.0000001

def compute_loss( weight):
    prediction = neuron(input, [weight],0.5)
    loss_value = loss(prediction,0.8)
    return loss_value


learning_rate = 1


for i in range(50):
    gradient = (compute_loss(weight+h) - compute_loss(weight)) / h  
    weight = weight - learning_rate * gradient
    print("step:", i, "Weight:", weight, "Loss:", compute_loss(weight)) 