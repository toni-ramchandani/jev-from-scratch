"""Preflight only: independent Torch gradients/Adam, Decimal, A/A and 50 train steps.

Never opens optdigits.tes and never computes held-out outcomes.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[2].parent/'digits-numerical-output'
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import torch
import torch.nn.functional as F
import mlp

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
DEFAULT_DATA = DEFAULT_OUT/'data'


def source_hashes():
    folder = Path(__file__).resolve().parent
    result = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
              for name in ('mlp.py','train.py','validate.py','fetch_data.py')}
    assert json.loads((folder/'protocol.json').read_text(encoding='utf-8'))['training']==mlp.CONFIG
    result['protocol'] = hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()
    return result


def torch_loss(parameters, x, digit, parity, detached):
    h = F.relu(x @ parameters['w1'] + parameters['b1'])
    h = F.relu(h @ parameters['w2'] + parameters['b2'])
    logits = h @ parameters['wd'] + parameters['bd']
    z = (h.detach() if detached else h) @ parameters['wp'] + parameters['bp']
    return F.cross_entropy(logits, digit) + F.binary_cross_entropy_with_logits(z, parity)


def compare(a, b, atol, rtol):
    a, b = np.asarray(a), np.asarray(b)
    error = np.abs(a-b)
    scale = atol + rtol*np.abs(b)
    return dict(pass_=bool(np.all(error <= scale)), max_abs=float(np.max(error)),
                max_scaled_error=float(np.max(error/scale)), atol=atol, rtol=rtol)


def gradient_check(p32, batch, dtype, detached):
    p = mlp.clone(p32, dtype)
    x32, y, parity32 = batch
    x, parity = x32.astype(dtype), parity32.astype(dtype)
    td = torch.float64 if dtype == np.float64 else torch.float32
    tp = {k: torch.tensor(v.copy(), dtype=td, requires_grad=True) for k,v in p.items()}
    tl = torch_loss(tp, torch.tensor(x), torch.tensor(y), torch.tensor(parity), detached)
    tl.backward()
    atol, rtol = (1e-10,1e-9) if dtype == np.float64 else (3e-6,3e-5)
    checks = {}
    for expression in ('stable','direct'):
        loss, grads, cache = mlp.loss_and_grad(p,x,y,parity,expression,detached)
        checks[expression] = {'loss': compare(loss,tl.detach().numpy(),atol,rtol),
                              'gradients': {k: compare(grads[k],tp[k].grad.numpy(),atol,rtol)
                                            for k in mlp.NAMES}}
    minimum = float(min(np.min(np.abs(cache['a1'])),np.min(np.abs(cache['a2']))))
    return dict(dtype=np.dtype(dtype).name, detached=detached,
                min_abs_relu_preactivation=minimum, no_relu_kink=minimum>1e-8,
                checks=checks,
                all_pass=minimum>1e-8 and all(c['loss']['pass_'] and
                    all(q['pass_'] for q in c['gradients'].values()) for c in checks.values()))


def adam_check(p32, state32, batch, dtype):
    p = mlp.clone(p32,dtype)
    state = dict(t=state32['t'], m=mlp.clone(state32['m'],dtype), v=mlp.clone(state32['v'],dtype))
    x,y,parity = batch
    _, grads, _ = mlp.loss_and_grad(p,x.astype(dtype),y,parity.astype(dtype))
    td = torch.float64 if dtype == np.float64 else torch.float32
    tp = {k:torch.tensor(v.copy(),dtype=td,requires_grad=True) for k,v in p.items()}
    dt = np.dtype(dtype).type
    optimizer = torch.optim.Adam(list(tp.values()), lr=float(dt(mlp.CONFIG['lr'])),
        betas=tuple(float(dt(b)) for b in mlp.CONFIG['betas']), eps=float(dt(mlp.CONFIG['eps'])),
        weight_decay=0, foreach=False, fused=False)
    for name, param in tp.items():
        optimizer.state[param] = dict(step=torch.tensor(float(state['t'])),
            exp_avg=torch.tensor(state['m'][name].copy()),
            exp_avg_sq=torch.tensor(state['v'][name].copy()))
        # Identical supplied gradients isolate the optimizer from framework VJP differences.
        param.grad = torch.tensor(grads[name].copy())
    mlp.adam_step(p,grads,state)
    optimizer.step()
    atol,rtol = (1e-12,1e-10) if dtype == np.float64 else (2e-7,2e-5)
    checks = {}
    for name,param in tp.items():
        checks[name] = dict(parameter=compare(p[name],param.detach().numpy(),atol,rtol),
                           m=compare(state['m'][name],optimizer.state[param]['exp_avg'].numpy(),atol,rtol),
                           v=compare(state['v'][name],optimizer.state[param]['exp_avg_sq'].numpy(),atol,rtol))
    return dict(dtype=np.dtype(dtype).name, supplied_gradients_identical=True,
                initial_t=state32['t'], checks=checks,
                all_pass=all(v['pass_'] for c in checks.values() for v in c.values()))


def decimal_check(p32, batch, dtype):
    p = mlp.clone(p32,dtype)
    x,y,parity = batch
    x,parity = x.astype(dtype),parity.astype(dtype)
    z = mlp.forward(p,x)['parity']
    n = len(x)
    with localcontext() as ctx:
        ctx.prec = 80
        reference = [(Decimal(1)/(Decimal(1)+(-Decimal.from_float(float(v))).exp())-
                     Decimal.from_float(float(t)))/Decimal(n)
                     for v,t in zip(z.ravel(),parity.ravel())]
    result = {}
    for expression in ('stable','direct'):
        g = ((mlp.sigmoid(z,expression)-parity)/np.dtype(dtype).type(n)).ravel()
        errors = [float(abs(Decimal.from_float(float(v))-r)) for v,r in zip(g,reference)]
        bound = float(8*np.finfo(dtype).eps/n)
        result[expression] = dict(max_abs_error=max(errors), mean_abs_error=float(np.mean(errors)),
                                  absolute_bound=bound, pass_=bool(max(errors)<=bound))
    return dict(dtype=np.dtype(dtype).name, precision_digits=80, n=n,
                logit_min=float(z.min()), logit_max=float(z.max()), checks=result,
                all_pass=all(c['pass_'] for c in result.values()))


def exact_state(a,b,sa,sb):
    return sa['t']==sb['t'] and all(np.array_equal(a[k],b[k]) and
        np.array_equal(sa['m'][k],sb['m'][k]) and np.array_equal(sa['v'][k],sb['v'][k]) for k in mlp.NAMES)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT/'preflight.json')
    args=parser.parse_args()
    result = dict(all_pass=False, role='Preflight; training data only; no held-out evaluation',
                  source_hashes=source_hashes(), python=platform.python_version(),
                  numpy=np.__version__, torch=torch.__version__,
                  dependency_resolution='Installed environment; no workspace dependency injection')
    try:
        x,y,parity=mlp.load_split(args.data_dir/'optdigits.tra','tra')
        result['training_file_sha256']=hashlib.sha256((args.data_dir/'optdigits.tra').read_bytes()).hexdigest()
        expected=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))['dataset']['files']['optdigits.tra']['sha256']
        assert result['training_file_sha256']==expected, 'Training data differs from official pinned bytes'
        result['training_array_sha256']=mlp.hash_arrays(dict(x=x,y=y,parity=parity))
        p=mlp.initialize(0); state=mlp.new_optimizer(p); order=mlp.permutations(0)
        result['initial_parameter_sha256']=mlp.hash_arrays(p)
        result['permutation_sha256']=mlp.hash_arrays(dict(permutations=order))
        validation_batch=(x[:16],y[:16],parity[:16])
        initial=[]
        for dtype in (np.float64,np.float32):
            initial.extend([gradient_check(p,validation_batch,dtype,False),
                            gradient_check(p,validation_batch,dtype,True),
                            adam_check(p,state,validation_batch,dtype),
                            decimal_check(p,validation_batch,dtype)])
        result['initial_reference_checks']=initial
        assert all(c['all_pass'] for c in initial), 'initial framework/scalar gate failed'
        # A/A before the representative 50-update training-only state.
        a,b=mlp.clone(p),mlp.clone(p); sa,sb=mlp.new_optimizer(a),mlp.new_optimizer(b)
        aa=[]
        for step,(_,rows) in enumerate(mlp.batches(order),1):
            for model,optimizer in ((a,sa),(b,sb)):
                _,g,_=mlp.loss_and_grad(model,x[rows],y[rows],parity[rows])
                mlp.adam_step(model,g,optimizer)
            aa.append(exact_state(a,b,sa,sb))
            if step==20:break
        result['aa20']=dict(exact_every_update=all(aa), updates=20, state_checks=aa)
        assert all(aa),'A/A execution gate failed'
        for step,(_,rows) in enumerate(mlp.batches(order),1):
            _,g,_=mlp.loss_and_grad(p,x[rows],y[rows],parity[rows])
            mlp.adam_step(p,g,state)
            if step==50:break
        trained=[]
        for dtype in (np.float64,np.float32):
            trained.extend([gradient_check(p,validation_batch,dtype,False),
                            gradient_check(p,validation_batch,dtype,True),
                            adam_check(p,state,validation_batch,dtype),
                            decimal_check(p,validation_batch,dtype)])
        result['trained50_reference_checks']=trained
        result['trained_state_parameter_sha256']=mlp.hash_arrays(p)
        result['dtype_assertions']='Every floating cache, gradient, parameter, Adam m/v/update checked; training FP32'
        result['all_pass']=all(c['all_pass'] for c in trained)
        assert result['all_pass'],'trained-state reference gate failed'
    except Exception as exc:
        result['error']=str(exc);result['traceback']=traceback.format_exc()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_pass=result['all_pass'],out=str(args.out),error=result.get('error'))))
    return 0 if result['all_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
