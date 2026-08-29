from task1_value import value, draw_dot

"""h = 0.0001


x1 = value(2.0)
x2 = value(0.0)
w1 = value(-3.0)
w2 = value(1.0)
b = value(6.8813735870195432)
n = x1*w1 + x2*w2 + b
o1 = n.tanh()


x1_ = value(2.0)
x2_ = value(0.0)
w1_ = value(-3.0+h)
w2_ = value(1.0)
b = value(6.8813735870195432)
n2 = x1_*w1_ + x2_*w2_ + b
o2 = n2.tanh()

grad_ = (o2.data - o1.data) / h
print(grad_)"""   


#x1=-1.5003564908333544
#x2= 0.49997742244678456
#w1 =0.9998841493119581
#w2=0.0


import torch

x1_t = torch.Tensor([2.0]).double(); x1_t.requires_grad = True
x2_t = torch.Tensor([0.0]).double(); x2_t.requires_grad = True
w1_t = torch.Tensor([-3.0]).double(); w1_t.requires_grad = True
w2_t = torch.Tensor([1.0]).double(); w2_t.requires_grad = True
b_t = torch.Tensor([6.8813735870195432]).double(); b_t.requires_grad = True

n_t = x1_t*w1_t + x2_t*w2_t + b_t
o_t = torch.tanh(n_t)

print(o_t.data.item())   

o_t.backward()
print(x1_t.grad.item())  
print(w1_t.grad.item())   
print(x2_t.grad.item())  
print(w2_t.grad.item())