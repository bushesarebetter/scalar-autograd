import argparse
import math
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, children=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = children
        self._op = op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= out.grad * self.data / other.data**2

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data**k, (self,), f"**{k:g}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # iterative post-order DFS; deep graphs (long sums) would blow the recursion limit
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((c, False) for c in node._prev if c not in seen)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        return topo


class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=i < len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def svm_loss(model, X, y, alpha=1e-4):
    scores = [model(list(map(Value, xi))) for xi in X]
    hinge = sum((1 - float(yi) * s).relu() for yi, s in zip(y, scores)) / len(X)
    reg = alpha * sum(p * p for p in model.parameters())
    acc = sum((s.data > 0) == (yi > 0) for yi, s in zip(y, scores)) / len(X)
    return hinge + reg, acc










