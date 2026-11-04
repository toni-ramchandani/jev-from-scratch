"""Small explicit-precision MLP; FP64 is supported only for reference checks."""
import hashlib
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

CONFIG = dict(hidden=[64, 32], seeds=[0, 1, 2], epochs=50, batch_size=128,
              lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
              probe_updates=[0, 1, 10, 100, 500, 1500], parity_weight=1,
              init_rng='numpy.default_rng(seed)',
              permutation_rng='numpy.default_rng(1000+seed)',
              dtype='float32', final_partial_batch=111)
NAMES = ('w1', 'b1', 'w2', 'b2', 'wd', 'bd', 'wp', 'bp')
EXPECTED_COUNTS = {
    'tra': [376,389,380,389,387,376,377,387,380,382],
    'tes': [178,182,177,183,181,182,181,179,174,180]}


def hash_arrays(arrays):
    h = hashlib.sha256()
    for key, value in arrays.items():
        a = np.ascontiguousarray(value)
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_split(path, split):
    raw = np.loadtxt(path, delimiter=',', dtype=np.int64)
    n = 3823 if split == 'tra' else 1797
    assert raw.shape == (n, 65), raw.shape
    assert np.all((raw[:, :64] >= 0) & (raw[:, :64] <= 16))
    assert np.all((raw[:, 64] >= 0) & (raw[:, 64] <= 9))
    y = np.ascontiguousarray(raw[:, 64])
    assert np.bincount(y, minlength=10).tolist() == EXPECTED_COUNTS[split]
    x = np.ascontiguousarray(raw[:, :64].astype(np.float32) / np.float32(16))
    parity = np.ascontiguousarray((y % 2).astype(np.float32).reshape(-1, 1))
    return x, y, parity


def initialize(seed):
    rng = np.random.default_rng(seed)
    p = {}
    for name, fan_in, fan_out, numerator in (
            ('1',64,64,2), ('2',64,32,2), ('d',32,10,1), ('p',32,1,1)):
        scale = np.sqrt(np.float32(numerator) / np.float32(fan_in))
        p['w'+name] = rng.standard_normal((fan_in, fan_out), dtype=np.float32) * scale
        p['b'+name] = np.zeros(fan_out, dtype=np.float32)
    p = {name: p[name] for name in NAMES}
    assert_float_arrays(p, np.float32)
    return p


def permutations(seed):
    rng = np.random.default_rng(1000 + seed)
    return np.stack([rng.permutation(3823) for _ in range(CONFIG['epochs'])])


def batches(order):
    for epoch, rows in enumerate(order):
        for start in range(0, len(rows), CONFIG['batch_size']):
            yield epoch, rows[start:start+CONFIG['batch_size']]


def clone(p, dtype=None):
    return {k: v.astype(dtype, copy=True) if dtype else v.copy() for k,v in p.items()}


def assert_float_arrays(arrays, dtype):
    for key, value in arrays.items():
        assert isinstance(value, (np.ndarray, np.generic)), (key, type(value))
        assert value.dtype == np.dtype(dtype), (key, value.dtype, dtype)
        assert np.all(np.isfinite(value)), ('nonfinite', key)


def sigmoid(z, expression):
    dt = z.dtype.type
    if expression == 'stable':
        e = np.exp(-np.abs(z))
        result = np.where(z >= dt(0), dt(1)/(dt(1)+e), e/(dt(1)+e))
    elif expression == 'direct':
        # Keep overflow visible. The resulting zero sigmoid is finite, but the
        # overflow event is separately counted in trace diagnostics.
        with np.errstate(over='ignore'):
            result = dt(1)/(dt(1)+np.exp(-z))
    else:
        raise ValueError(expression)
    assert result.dtype == z.dtype
    return result


def forward(p, x):
    dtype = p['w1'].dtype
    dt = dtype.type
    assert_float_arrays(p, dtype)
    assert x.dtype == dtype
    a1 = x @ p['w1'] + p['b1']
    h1 = np.maximum(a1, dt(0))
    a2 = h1 @ p['w2'] + p['b2']
    h2 = np.maximum(a2, dt(0))
    d = h2 @ p['wd'] + p['bd']
    z = h2 @ p['wp'] + p['bp']
    cache = dict(a1=a1, h1=h1, a2=a2, h2=h2, digit=d, parity=z)
    assert_float_arrays(cache, dtype)
    return cache


def loss_and_grad(p, x, y, parity, expression='stable', detached=False):
    c = forward(p, x)
    dtype = x.dtype
    dt = dtype.type
    assert parity.dtype == dtype
    n = dt(len(x))
    scores, z = c['digit'], c['parity']
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    ex = np.exp(shifted)
    denom = np.sum(ex, axis=1, keepdims=True, dtype=dtype)
    prob = ex / denom
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(x)), y] = dt(1)
    digit_loss = np.sum(np.log(denom[:,0]) - shifted[np.arange(len(x)), y], dtype=dtype) / n
    parity_loss = np.sum(np.maximum(z, dt(0)) - z*parity + np.log1p(np.exp(-np.abs(z))), dtype=dtype) / n
    total = digit_loss + parity_loss
    dd = (prob-onehot)/n
    sig = sigmoid(z, expression)
    dp = (sig-parity)/n
    grads = dict(wd=c['h2'].T @ dd, bd=np.sum(dd, axis=0, dtype=dtype),
                 wp=c['h2'].T @ dp, bp=np.sum(dp, axis=0, dtype=dtype))
    dh2 = dd @ p['wd'].T
    if not detached:
        dh2 = dh2 + dp @ p['wp'].T
    da2 = dh2 * (c['a2'] > dt(0))
    grads['w2'] = c['h1'].T @ da2
    grads['b2'] = np.sum(da2, axis=0, dtype=dtype)
    da1 = (da2 @ p['w2'].T) * (c['a1'] > dt(0))
    grads['w1'] = x.T @ da1
    grads['b1'] = np.sum(da1, axis=0, dtype=dtype)
    grads = {name: grads[name] for name in NAMES}
    assert_float_arrays(grads, dtype)
    assert_float_arrays(dict(total=total, digit_loss=digit_loss, parity_loss=parity_loss,
                             shifted=shifted, ex=ex, denominator=denom, prob=prob,
                             onehot=onehot, dd=dd, sig=sig, dp=dp,
                             dh2=dh2, da2=da2, da1=da1), dtype)
    c.update(digit_loss=digit_loss, parity_loss=parity_loss, binary_gradient=dp)
    return total, grads, c


def new_optimizer(p):
    return dict(t=0, m={k: np.zeros_like(v) for k,v in p.items()},
                v={k: np.zeros_like(v) for k,v in p.items()})


def adam_step(p, grads, state):
    dtype = p['w1'].dtype
    dt = dtype.type
    one = dt(1)
    b1, b2 = (dt(x) for x in CONFIG['betas'])
    lr, eps = dt(CONFIG['lr']), dt(CONFIG['eps'])
    assert_float_arrays(p, dtype); assert_float_arrays(grads, dtype)
    assert_float_arrays(state['m'], dtype); assert_float_arrays(state['v'], dtype)
    state['t'] += 1
    # Explicit dtype on powers prevents integer/float promotion in bias correction.
    corr1 = one - np.power(b1, state['t'], dtype=dtype)
    corr2 = one - np.power(b2, state['t'], dtype=dtype)
    for name in NAMES:
        m, v, g = state['m'][name], state['v'][name], grads[name]
        m *= b1; m += (one-b1)*g
        v *= b2; v += (one-b2)*(g*g)
        mh, vh = m/corr1, v/corr2
        update = lr*mh/(np.sqrt(vh)+eps)
        assert_float_arrays(dict(mh=mh, vh=vh, update=update), dtype)
        p[name] -= update
    assert_float_arrays(p, dtype)
    assert_float_arrays(state['m'], dtype); assert_float_arrays(state['v'], dtype)


def checkpoint(path, p, state):
    arrays = {f'p_{k}': v for k,v in p.items()}
    arrays.update({f'm_{k}': v for k,v in state['m'].items()})
    arrays.update({f'v_{k}': v for k,v in state['v'].items()})
    arrays['t'] = np.array(state['t'], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def evaluation_predictions(p, x):
    raw = forward(p, x)
    d, z = raw['digit'].astype(np.float64), raw['parity'][:,0].astype(np.float64)
    shift = d-np.max(d, axis=1, keepdims=True)
    logp = shift - np.log(np.sum(np.exp(shift), axis=1, keepdims=True))
    return dict(digit_logits=raw['digit'], parity_logits=raw['parity'][:,0],
                digit_prob=np.exp(logp), digit_logprob=logp,
                parity_prob=sigmoid(z, 'stable'))


def quality(pred, y):
    p, z, q = pred['digit_prob'], pred['parity_logits'].astype(np.float64), pred['parity_prob']
    onehot = np.eye(10, dtype=np.float64)[y]
    parity = (y % 2).astype(np.float64)
    return dict(digit_nll=float(-np.mean(pred['digit_logprob'][np.arange(len(y)),y])),
                digit_brier=float(np.mean(np.sum((p-onehot)**2, axis=1))),
                digit_accuracy=float(np.mean(np.argmax(p, axis=1)==y)),
                parity_nll=float(np.mean(np.maximum(z,0)-z*parity+np.log1p(np.exp(-np.abs(z))))),
                parity_brier=float(np.mean((q-parity)**2)),
                parity_accuracy=float(np.mean((q >= 0.5)==parity)))


def distances(a, b):
    pa, pb = a['digit_prob'], b['digit_prob']
    dis = np.argmax(pa,axis=1) != np.argmax(pb,axis=1)
    qa, qb = a['parity_prob'], b['parity_prob']
    pdis = (qa >= 0.5) != (qb >= 0.5)
    margins = lambda p: np.sort(p, axis=1)[:,-1] - np.sort(p, axis=1)[:,-2]
    return dict(n=len(pa), digit_mean_tv=float(np.mean(0.5*np.sum(np.abs(pa-pb), axis=1))),
                digit_max_probability_difference=float(np.max(np.abs(pa-pb))),
                digit_argmax_disagreements=int(np.sum(dis)), digit_disagreement_rate=float(np.mean(dis)),
                digit_disagreement_margins_a=margins(pa)[dis].tolist(),
                digit_disagreement_margins_b=margins(pb)[dis].tolist(),
                parity_mean_tv=float(np.mean(np.abs(qa-qb))),
                parity_max_probability_difference=float(np.max(np.abs(qa-qb))),
                parity_threshold_disagreements=int(np.sum(pdis)),
                parity_disagreement_rate=float(np.mean(pdis)))
