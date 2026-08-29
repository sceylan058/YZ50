class value:
    def __init__(self, data,children=(), _op="",label=""):
        self.data = data
        self.grad = 0
        self._prev = set(children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
    
    def __repr__(self):
        return f"value(data = {self.data})"
    
    def __add__(self, other):
        
        other = other if isinstance(other, value) else value(other)
        out=value(self.data + other.data,(self,other), "+")
        def _backward():
            self.grad += 1.0* out.grad
            other.grad += 1.0* out.grad
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data * other.data,(self,other),"*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

#burası task 2te açılacak    

    
    def tanh(self):
        x = self.data
        e = 2.718
        t = (e**(2*x) - 1) / (e**(2*x) + 1)
        out = value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1-t**2)*out.grad
        out._backward = _backward
        return out
        
        
        
#burası task 4 te açılacak 
        
    

    def exp(self):
        x = self.data
        e = 2.718
        t =e**x
        out = value(t,(self,),"exp")
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward    
        return out
    
    def __rmul__(self, other):
        return self * other
        
    def __neg__(self):
        t = self *-1
        return t
        
    def __sub__(self,other):
        t = self + (-other)
        return t 
        
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other ):
        t = self * other** -1
        return t
        
        
        #3. task
        
                
    def backward(self):
        topo=[]
        visit= set()
        self.grad= 1.0
        def build_topo(m):
            if m not in visit:
                visit.add(m)
                for child in m._prev:
                    build_topo(child)
                topo.append(m)
        build_topo(self)    
        
    
        for node in reversed(topo):
            node._backward()




from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
    
"""if __name__ == "__main__":  
    
    a = value(2, label='a')
    b = value(-5, label='b')
    c = value(4, label='c')
    
    d=a+b+c 
    d.label= "d"
    print(d.data)
    print(d._prev)
      
    dot = draw_dot(d)
    dot.render('graph', view=True)"""
    
