"""autograd.py: a tiny tensor-level reverse-mode autodiff engine in NumPy.

Same idea as Karpathy's micrograd, but each node holds an array rather than a
scalar, so a transformer trains in minutes on a laptop CPU. Every op below has
a finite-difference gradient check in tests.py.
"""
import numpy as np

DTYPE = np.float32  # tests switch this to float64 for gradient checks


def set_dtype(dtype):
    global DTYPE
    DTYPE = dtype


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, parents=(), requires_grad=False):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        self._parents = parents
        self._backward = None
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    shape = property(lambda self: self.data.shape)

    def _acc(self, g):
        if self.requires_grad:
            self.grad = g if self.grad is None else self.grad + g

    def backward(self):
        order, seen = [], set()

        def visit(t):
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._backward is not None and t.grad is not None:
                t._backward(t.grad)

    # ---- elementwise -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        def backward(g):
            if self.requires_grad:
                self._acc(_unbroadcast(g, self.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(g, other.shape))
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        def backward(g):
            if self.requires_grad:
                self._acc(_unbroadcast(g * other.data, self.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(g * self.data, other.shape))
        out._backward = backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def backward(g):
            self._acc(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            other._acc(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))
        out._backward = backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    # ---- shape -------------------------------------------------------------
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))
        out._backward = lambda g: self._acc(g.reshape(self.shape))
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,))
        inverse = np.argsort(axes)
        out._backward = lambda g: self._acc(g.transpose(*inverse))
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())
        out._backward = backward
        return out


# ---- neural-network ops (fused where that makes the backward pass simpler) ----

def gelu(x):
    """GELU, tanh approximation (the variant used in GPT-2)."""
    k = float(np.sqrt(2.0 / np.pi))  # a Python float: a NumPy float64 would upcast float32 arrays
    xd = x.data
    x2 = xd * xd
    t = np.tanh(k * xd * (1.0 + 0.044715 * x2))
    out = Tensor(0.5 * xd * (1.0 + t), (x,))
    def backward(g):
        dinner = k * (1.0 + 3 * 0.044715 * x2)
        x._acc(g * (0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t * t) * dinner))
    out._backward = backward
    return out


def softmax(x, axis=-1):
    z = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(y, (x,))
    out._backward = lambda g: x._acc(y * (g - (g * y).sum(axis=axis, keepdims=True)))
    return out


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x.data - mu) * inv
    out = Tensor(xhat * gamma.data + beta.data, (x, gamma, beta))
    def backward(g):
        n = x.shape[-1]
        dxhat = g * gamma.data
        x._acc(inv / n * (n * dxhat - dxhat.sum(-1, keepdims=True)
                          - xhat * (dxhat * xhat).sum(-1, keepdims=True)))
        gamma._acc(_unbroadcast(g * xhat, gamma.shape))
        beta._acc(_unbroadcast(g, beta.shape))
    out._backward = backward
    return out


def take_rows(table, idx):
    """table[idx] for a 2-D table: embedding lookup and readout gathering.
    Backward scatters gradients back; a one-hot matmul is much faster than
    np.add.at and handles repeated indices the same way."""
    idx = np.asarray(idx)
    out = Tensor(table.data[idx], (table,))
    def backward(g):
        flat = idx.reshape(-1)
        onehot = np.zeros((flat.size, table.shape[0]), dtype=g.dtype)
        onehot[np.arange(flat.size), flat] = 1.0
        table._acc(onehot.T @ g.reshape(flat.size, -1))
    out._backward = backward
    return out


def index0(x, i):
    """x[i] along the first axis (used to split fused q, k, v)."""
    out = Tensor(x.data[i], (x,))
    def backward(g):
        full = np.zeros_like(x.data)
        full[i] = g
        x._acc(full)
    out._backward = backward
    return out


def grouped_softmax_xent(scores, group, target, n_groups):
    """Cross-entropy (the log score) of a softmax taken *within* each group.

    scores: (N,) one score per option; group: (N,) question id of each option;
    target: (N,) 1.0 for the observed outcome of its question, else 0.
    Returns the mean over questions. The gradient is softmax - onehot.
    """
    s = scores.data
    m = np.full(n_groups, -np.inf, dtype=s.dtype)
    np.maximum.at(m, group, s)
    e = np.exp(s - m[group])
    z = np.zeros(n_groups, dtype=s.dtype)
    np.add.at(z, group, e)
    p = e / z[group]
    if n_groups == 0:
        out = Tensor(0.0, (scores,))
        out._backward = lambda g: scores._acc(np.zeros_like(s))
        return out
    # Log-sum-exp keeps the loss and derivative consistent at extreme logits.
    log_p = s - m[group] - np.log(z[group])
    loss = -(log_p * target).sum() / n_groups
    out = Tensor(loss, (scores,))
    out._backward = lambda g: scores._acc(g * (p - target) / n_groups)
    return out


def group_softmax_np(s, group, n_groups):
    """Same softmax-within-group, NumPy only, for inference."""
    m = np.full(n_groups, -np.inf)
    np.maximum.at(m, group, s)
    e = np.exp(s - m[group])
    z = np.zeros(n_groups)
    np.add.at(z, group, e)
    return e / z[group]


def bce_with_logits(logits, y):
    """Binary log score on logits, numerically stable."""
    z = logits.data
    if z.size == 0:
        out = Tensor(0.0, (logits,))
        out._backward = lambda g: logits._acc(np.zeros_like(z))
        return out
    loss = (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean()
    out = Tensor(loss, (logits,))
    sig = sigmoid_np(z)
    out._backward = lambda g: logits._acc(g * (sig - y) / z.size)
    return out


def sigmoid_np(z):
    """Stable sigmoid for either sign of a finite logit."""
    z = np.asarray(z)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter), from scratch."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, decay=()):
        self.params, self.lr, self.betas, self.eps = params, lr, betas, eps
        self.decay = set(id(p) for p in decay)  # only these get weight decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self, wd=0.01):
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m *= b1; m += (1 - b1) * p.grad
            v *= b2; v += (1 - b2) * p.grad ** 2
            mhat = m / (1 - b1 ** self.t)
            vhat = v / (1 - b2 ** self.t)
            if id(p) in self.decay:
                p.data -= self.lr * wd * p.data
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
