import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

class Value:

    def __init__(self, data, _children = (), _op = '', label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self): #把这个对象丢进print会变成什么样子
        return f"Value(data={self.data})"

    # 正常的加法运算
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backwards():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backwards
        return out

    # 处理负号
    def __neg__(self):
        return self * -1

    # 减法运算，因为减法可以看作加上一个负数，所以直接调用加法和负号的实现
    def __sub__(self, other):
        return self + (-other)

    # mul处理数字在右边的情况
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backwards():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backwards

        return out

    # 幂运算
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backwards():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backwards

        return out
    
    # rmul处理数字在左边的情况
    def __rmul__(self, other):
        return self * other

    # 除法运算
    def __truediv__(self, other):
        return self * other**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backwards():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backwards

        return out

    # 指数函数
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backwards():
            self.grad += out.data * out.grad
        out._backward = _backwards

        return out

    def backwards(self):
        #得到这个神经网络的拓扑排序
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        print(topo)

        self.grad = 1
        for node in reversed(topo):#逆topo顺序进行梯度传播
            node._backward()

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
    dot = Digraph(format = 'svg', graph_attr = {'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape = 'record')
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

if __name__ == "__main__":
    x1 = Value(2.0, label = 'x1')
    x2 = Value(0.0, label = 'x2')
    w1 = Value(-3.0, label = 'w1')
    w2 = Value(1.0, label = 'w2')
    b = Value(6.8813735870195432, label = 'b')
    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    # ---
    e = (2*n).exp()
    o = (e - 1)/(e + 1)
    # ---
    o.label = 'o'
    o.grad = 1
    o.backwards()
    dot = draw_dot(o)
    dot.render('graph')

