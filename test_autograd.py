import math
import random

import pytest
import torch

from autograd import MLP, Value

# Unary ops are wrapped so every random expression stays inside its domain.
UNARY = [
    lambda a: a.tanh(),
    lambda a: a.relu(),
    lambda a: (a * 0.3).exp(),
    lambda a: (a * a + 1).log(),
    lambda a: (a * a + 0.5) ** 1.5,
    lambda a: -a,
]
BINARY = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
    lambda a, b: a / (b * b + 1),
    lambda a, b: 2.0 - a * b,
    lambda a, b: 3.0 / (a * a + 1) + b,
]


def random_expr(rng, xs, depth):
    if depth == 0:
        return rng.choice(xs)
    if rng.random() < 0.4:
        return rng.choice(UNARY)(random_expr(rng, xs, depth - 1))
    return rng.choice(BINARY)(random_expr(rng, xs, depth - 1), random_expr(rng, xs, depth - 1))


@pytest.mark.parametrize("seed", range(30))
def test_gradients_match_finite_differences(seed):
    rng = random.Random(seed)
    vals = [rng.uniform(-2, 2) for _ in range(4)]

    def f(v):
        xs = [Value(a) for a in v]
        return random_expr(random.Random(seed + 1000), xs, 4), xs

    out, xs = f(vals)
    out.backward()
    eps = 1e-6
    for i in range(len(vals)):
        up, down = list(vals), list(vals)
        up[i] += eps
        down[i] -= eps
        numeric = (f(up)[0].data - f(down)[0].data) / (2 * eps)
        assert math.isclose(xs[i].grad, numeric, rel_tol=1e-4, abs_tol=1e-6), (i, xs[i].grad, numeric)


def test_matches_torch():
    a, b, c = Value(-4.0), Value(2.0), Value(0.7)
    d = a * b + b**3 - (a / b).tanh()
    e = (c * d).relu() + (d * d + 1).log() - (c - a).exp() / 10
    f = e * e / 3.0 + 1.0 / (c + 2) - a.relu()
    f.backward()

    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (-4.0, 2.0, 0.7))
    td = ta * tb + tb**3 - (ta / tb).tanh()
    te = (tc * td).relu() + (td * td + 1).log() - (tc - ta).exp() / 10
    tf = te * te / 3.0 + 1.0 / (tc + 2) - ta.relu()
    tf.backward()

    assert math.isclose(f.data, tf.item(), rel_tol=1e-9)
    for v, t in [(a, ta), (b, tb), (c, tc)]:
        assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-9, abs_tol=1e-12)


def test_mlp_matches_torch():
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    x = [0.5, -1.2, 2.0]
    out = model([Value(v) for v in x]).tanh()
    out.backward()

    params = model.parameters()
    tp = [torch.tensor(p.data, dtype=torch.float64, requires_grad=True) for p in params]
    it = iter(tp)
    h = [torch.tensor(v, dtype=torch.float64) for v in x]
    for li, layer in enumerate(model.layers):
        nxt = []
        for n in layer.neurons:
            w = [next(it) for _ in n.w]
            act = sum((wi * hi for wi, hi in zip(w, h)), next(it))
            nxt.append(act.relu() if n.nonlin else act)
        h = nxt
    h[0].tanh().backward()

    for p, t in zip(params, tp):
        assert math.isclose(p.grad, t.grad.item(), rel_tol=1e-9, abs_tol=1e-12)


def test_shared_node_accumulates():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0
