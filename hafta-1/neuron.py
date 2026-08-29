import math

def sigmoid(z):
    result = 1 / (1 + math.exp(-z))
    return result
    
inputs = [0.5, 0.2, 0.1]
weights = [0.4, 0.3, 0.9]

    
def neuron(inputs, weights, bias):
    z = 0
    for i in range(len(inputs)):
        z += inputs[i] * weights[i]
    z += bias
    return sigmoid(z)


weights_list = [
    [0.4, 0.3, 0.9],
    [0.2, 0.8, 0.5],
    [0.6, 0.1, 0.7]]
    
biases_list = [0.5, 0.1, 0.3]

def layer(inputs, weights_list, biases_list):
    outputs = []
    for i in range(len(weights_list)):
        output = neuron(inputs, weights_list[i], biases_list[i])
        outputs.append(output)
    return outputs

bias = 0.5
result = neuron(inputs, weights, bias)
print("Neuron output:", result)

layer_output = layer(inputs, weights_list, biases_list)
print("layer output:", layer_output)