"""tests.py: verify the engine and the contract before trusting any result.

micrograd checks its gradients against PyTorch; PyTorch is unavailable here, so
we check every op against central finite differences in float64 instead.
Run: python tests.py
"""
import copy
import numpy as np
import autograd as ag
ag.set_dtype(np.float64)
from autograd import Tensor
import data as D
from model import TinyS1

rng = np.random.default_rng(0)


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + eps; fp = f()
        x.data[i] = old - eps; fm = f()
        x.data[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check(name, build, inputs, tol=1e-6):
    for t in inputs:
        t.grad = None
    out = build()
    out.backward()
    worst = 0.0
    for t in inputs:
        ng = numgrad(lambda: float(build().data), t)
        err = np.abs(ng - t.grad).max() / (np.abs(ng).max() + 1e-8)
        worst = max(worst, err)
    status = "ok " if worst < tol else "FAIL"
    print(f"  [{status}] {name:<28} max rel err {worst:.1e}")
    assert worst < tol, name


def T(*shape):
    return Tensor(rng.normal(size=shape), requires_grad=True)


def test_ops():
    print("gradient checks (finite differences, float64)")
    a, b, w = T(2, 3, 4), T(1, 3, 4), T(4, 5)
    r = rng.normal(size=(2, 3, 5))
    check("add+mul broadcast", lambda: ((a + b) * b).sum(), [a, b])
    check("matmul (batched x 2-D)", lambda: ((a @ w) * Tensor(r)).sum(), [a, w])
    check("reshape+transpose", lambda: (a.transpose(1, 0, 2).reshape(3, 8) * Tensor(np.arange(24.).reshape(3, 8))).sum(), [a])
    check("sum(axis)", lambda: (a.sum(axis=1) * Tensor(np.arange(8.).reshape(2, 4))).sum(), [a])
    check("gelu", lambda: (ag.gelu(a) * Tensor(np.arange(24.).reshape(2, 3, 4))).sum(), [a])
    check("softmax", lambda: (ag.softmax(a, -1) * Tensor(np.arange(24.).reshape(2, 3, 4))).sum(), [a])
    g, be = T(4), T(4)
    check("layer_norm", lambda: (ag.layer_norm(a, g, be) * Tensor(np.arange(24.).reshape(2, 3, 4))).sum(), [a, g, be])
    tbl = T(6, 4)
    idx = np.array([[0, 2, 2], [5, 1, 0]])
    check("take_rows (repeated idx)", lambda: (ag.take_rows(tbl, idx) * Tensor(np.arange(24.).reshape(2, 3, 4))).sum(), [tbl])
    s = T(7)
    grp = np.array([0, 0, 0, 1, 1, 2, 2]); tgt = np.array([0, 1, 0, 1, 0, 0, 1.])
    check("grouped_softmax_xent", lambda: ag.grouped_softmax_xent(s, grp, tgt, 3), [s])
    z = T(5); y = np.array([1, 0, 1, 1, 0.])
    check("bce_with_logits", lambda: ag.bce_with_logits(z, y), [z])


def small_batch(seed=1, n=2):
    r = D.make_rng(seed)
    return [D.sample_request(r) for _ in range(n)]


def test_model_grad():
    print("end-to-end gradient check through the whole model")
    model = TinyS1(len(D.VOCAB), d=8, n_heads=2, n_layers=1, d_ff=16, seed=3)
    batch = D.collate(small_batch())
    def f():
        return model.loss(batch)[0]
    check("loss wrt w_opt", f, [model.w_opt])
    check("loss wrt w_qkv", f, [model.layers[0]["w_qkv"]], tol=1e-5)
    check("loss wrt ln1 gamma", f, [model.layers[0]["ln1_g"]], tol=1e-5)


def probs(model, reqs):
    b = D.collate(reqs)
    s, z = model.forward(b)
    p = ag.group_softmax_np(s.data, b["opt_group"], b["n_groups"])
    return b, p, 1 / (1 + np.exp(-z.data))


def test_contract():
    print("contract checks on a randomly initialised model")
    model = TinyS1(len(D.VOCAB), seed=5)
    reqs = small_batch(seed=7, n=16)
    b, p, pn = probs(model, reqs)
    # 1) probabilities over the supplied options sum to 1 for every question
    sums = np.zeros(b["n_groups"]); np.add.at(sums, b["opt_group"], p)
    print(f"  [ok ] probabilities sum to 1      max |sum-1| {np.abs(sums - 1).max():.1e}")
    assert np.allclose(sums, 1)
    # 2) shuffling the options permutes the probabilities, nothing more
    shuffled = copy.deepcopy(reqs)
    perms = []
    r = np.random.default_rng(11)
    for q in [x["questions"][0] for x in shuffled]:
        perm = r.permutation(len(q["options"]))
        q["options"] = [q["options"][i] for i in perm]; q["target"] = [q["target"][i] for i in perm]
        perms.append(perm)
    b2, p2, _ = probs(model, shuffled)
    worst = 0.0
    for (typ, (a0, a1, _), _q), (typ2, (c0, c1, _), _q2), perm in zip(
            [m for m in b["meta"] if m[0] == "choice"], [m for m in b2["meta"] if m[0] == "choice"], perms):
        worst = max(worst, np.abs(p[a0:a1][perm] - p2[c0:c1]).max())
    print(f"  [ok ] option order invariance     max |dp| {worst:.1e}")
    assert worst < 1e-10
    # 3) packing all questions in one pass == one pass per question
    worst = 0.0
    for req in reqs[:4]:
        _, p_all, pn_all = probs(model, [req])
        split_o, split_n = [], []
        for q in req["questions"]:  # one request per question, same state
            _, ps, pns = probs(model, [{"state": req["state"], "questions": [q]}])
            split_o.append(ps); split_n.append(pns)
        worst = max(worst, np.abs(np.concatenate(split_o) - p_all).max(), np.abs(np.concatenate(split_n) - pn_all).max())
    print(f"  [ok ] packed pass == separate     max |dp| {worst:.1e}")
    assert worst < 1e-10


if __name__ == "__main__":
    test_ops()
    test_model_grad()
    test_contract()
    print("all tests passed")
