import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random
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

    def __radd__(self, other): #处理数字在左边的情况
        return self + other

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
        #print(topo)

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

# 用random随机化初始权重和bias，后面通过训练更新权重
class Neuron:

    def __init__(self, nin):# nin: number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() # 非线性压缩
        return out

    # 收集各个神经元的参数
    def parameters(self):
        return self.w + [self.b]

# 一个Layer就是几个Neuron的汇聚
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# 多层感知机就是几个Layer的汇聚
class MLP:
    # 数组sz是表示每一层都有几个神经元
    # 比如[3,4,4,1]表示输入层有3个神经元，第一隐藏层有4个神经元，
    # 第二隐藏层有4个神经元，输出层有1个神经元
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    # 这三个层次的parameters函数都是为了收集所有的参数，方便后续进行梯度下降更新
    
if __name__ == "__main__":
    #a simple test
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    # print(n(x))
    # dot = draw_dot(n(x))
    # dot.render('graph')
    # xs是一个training dataset，里面有4个样本，每个样本有3个特征
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets理想的输出
    for k in range(20):
        ypred = [n(x) for x in xs]
        # loss function衡量预测的输出和理想输出之间的差距
        loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys))

        for p in n.parameters():
            p.grad = 0.0 #梯度清零(easy to forget)因为梯度是累加而不是单纯替代

        loss.backwards()
        # dot = draw_dot(loss)
        # dot.render('graph')
        for p in n.parameters():
            p.data += -0.05 * p.grad 
            #可以有不同的梯度更新函数，此处的0.01就是learning rate
        print(k, loss.data)