from task1_value import value, draw_dot
import math

a = value(2.0, label='a')
b = value(-3.0, label='b')
c = value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = value(-2.0, label='f')
L = d * f; L.label = 'L'
 
 
"""print(L)



L.grad = 1.0
f.grad = 4.0
d.grad = -2.0
e.grad = -2.0
c.grad = -2.0
b.grad = -4.0
a.grad = 6.0

# L.grad =  Dl/Dl = 1
# f.grad = d.data * dl/dl = d.data = d =4.0
# d.grad = f.data * dl/dl = f.data = f = -2.0
# e.grad = dd/de * dl/dd = 1* -2.0 = -2.0
# c.grad = dd/dc * dl/dd = 1* -2.0 = -2.0
# b.grad = de/db * dd/de * dl/dd = 2 * 1 * -2.0 = -4.0
# a.grad = de/da * dd/de * dl/dd = -3.0 * 1 * -2.0 = 6.0

dot = draw_dot(L)
dot.render('graph', view=True)"""





"""x1 = value(2.0, label='x1')
x2 = value(0.0, label='x2')
w1 = value(-3.0, label='w1')
w2 = value(1.0, label='w2')
b = value(6.8813735870195432, label='b')
 
x1w1 = x1 * w1; x1w1.label = 't1'
x2w2 = x2 * w2; x2w2.label = 't2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 't1+t2'

n = x1w1x2w2 + b ; n.label = 'n'

#task 4 ü anlatırken bu satırı kapat
o = n.tanh(); o.label = "o"


#e = (2*n).exp()
#o = (e - 1) / (e + 1)
#o.label= "o"
#o.backward()


# do/do = 1
# do/dn = 1- o**2 = 0.5
# dn/t1+t2= 1*0.5 = 0.5
# dn/b= 1*0.5 = 0.5
#dt1+t2/t1 = 1*0.5*1= 0.5
#dt1+t2/t2 = 1*0.5*1= 0.5
#dt1/dx1 = -3.0 *1*0.5*1 =-1.5
#dt2/dx2 = 1.0 *1*0.5*1 =0.5
#dt1/dw1 = 2.0 *1*0.5*1 =1
#dt2/dw2 = 0.0 *1*0.5*1 =0


o.grad = 1.0
n.grad = (1 - o.data**2) * o.grad

x1w1x2w2.grad = 1 * n.grad     # t1+t2
b.grad = 1 * n.grad

x1w1.grad = 1 * x1w1x2w2.grad  # t1
x2w2.grad = 1 * x1w1x2w2.grad  # t2

x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad


dot = draw_dot(o)
dot.render('graph', view=True)"""


"""L.backward()
print("a.grad:", a.grad)   
print("b.grad:", b.grad)   
print("c.grad:", c.grad)   
print("d.grad:", d.grad)   
print("e.grad:", e.grad)   
print("f.grad:", f.grad)"""