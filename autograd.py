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


def train_moons(steps=100, n=100, seed=1337):
    from sklearn.datasets import make_moons

    random.seed(seed)
    X, y = make_moons(n_samples=n, noise=0.1, random_state=seed)
    y = y * 2 - 1
    model = MLP(2, [16, 16, 1])
    for k in range(steps):
        loss, acc = svm_loss(model, X, y)
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        lr = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k:3d}  loss {loss.data:.4f}  acc {acc:.0%}")
    return model, X, y, acc


def plot_boundary(model, X, y, path):
    h = 0.1
    xs = np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, h)
    ys = np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, h)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.array([model([Value(a), Value(b)]).data for a, b in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.contourf(xx, yy, zz, levels=[zz.min() - 1, 0, zz.max() + 1], colors=["#fbe1d6", "#d6e6f8"])
    ax.contour(xx, yy, zz, levels=[0], colors=MUTED, linewidths=1)
    for cls, color, name in [(-1, ORANGE, "class 0"), (1, BLUE, "class 1")]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=30, c=color, edgecolors="white", linewidths=1, label=name)
    ax.set_title("MLP(2, [16, 16, 1]) on make_moons", color=INK)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_graph(root, path):
    topo = root.backward()
    depth = {}
    for node in topo:  # children come before parents in topo order
        depth[node] = 1 + max((depth[c] for c in node._prev), default=-1)

    # order rows by reverse DFS from the root so siblings stay together
    columns = {}
    for node in reversed(topo):
        columns.setdefault(depth[node], []).append(node)
    pos = {}
    tallest = max(len(c) for c in columns.values())
    for d, nodes in columns.items():
        for i, node in enumerate(nodes):
            pos[node] = (2 * d, (len(nodes) - 1) / 2 - i)

    ncols = 2 * max(columns) + 1
    fig, ax = plt.subplots(figsize=(1.25 * ncols + 1, 1.0 * tallest + 1))
    boxes = {}
    for node, (x, yv) in pos.items():
        name = node.label or ""
        text = f"{name}\ndata {node.data:.4f}\ngrad {node.grad:.4f}".lstrip()
        boxes[node] = ax.text(x, yv, text, ha="center", va="center", fontsize=8, family="monospace", color=INK,
                              bbox=dict(boxstyle="round,pad=0.4", fc="#d6e6f8", ec=BLUE, lw=1), zorder=3)
    for node in topo:
        if not node._prev:
            continue
        x, yv = pos[node]
        op = ax.text(x - 1, yv, node._op, ha="center", va="center", fontsize=9, color=INK,
                     bbox=dict(boxstyle="circle,pad=0.3", fc="#fbe1d6", ec=ORANGE, lw=1), zorder=3)
        arrow = dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=2, shrinkB=2)
        ax.annotate("", xy=(0, 0.5), xycoords=boxes[node], xytext=(1, 0.5), textcoords=op, arrowprops=arrow)
        for c in node._prev:
            ax.annotate("", xy=(0, 0.5), xycoords=op, xytext=(1, 0.5), textcoords=boxes[c], arrowprops=arrow)
    ax.set_xlim(-0.8, ncols - 0.2)
    ax.set_ylim(-tallest / 2, tallest / 2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_expression():
    x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
    w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")
    x1w1, x2w2 = x1 * w1, x2 * w2
    x1w1.label, x2w2.label = "x1*w1", "x2*w2"
    s = x1w1 + x2w2
    s.label = "x1w1+x2w2"
    n = s + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="20 steps, for a fast smoke run")
    args = ap.parse_args()

    draw_graph(example_expression(), "graph.png")
    print("saved graph.png")
    model, X, y, acc = train_moons(steps=20 if args.quick else args.steps)
    plot_boundary(model, X, y, "moons.png")
    print(f"final train accuracy {acc:.0%}, saved moons.png")
