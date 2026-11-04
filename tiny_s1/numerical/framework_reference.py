"""Independent PyTorch check of TinyS1 equations and AdamW.

No training-source mutation. FP64 tests mathematical agreement; FP32 comparisons
are numerical diagnostics, not assertions of cross-framework bit identity.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE))
import autograd as ag
import data as D
from model import TinyS1
from train import lr_at

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def named_parameters(model):
    names = ('tok', 'pos', 'seg', 'lnf_g', 'lnf_b', 'w_opt', 'w_noul', 'b_noul')
    result = {name:getattr(model, name) for name in names}
    for i, layer in enumerate(model.layers):
        result.update({f'layers.{i}.{key}':value for key,value in layer.items()})
    assert list(result.values()) == model.parameters()
    return result


def torch_parameters(model, dtype):
    return {name:torch.tensor(value.data.copy(), dtype=dtype, requires_grad=True)
            for name,value in named_parameters(model).items()}


def forward(parameters, batch, d, heads, layers):
    def integer(name): return torch.as_tensor(batch[name], dtype=torch.long)
    p = parameters
    h = p['tok'][integer('ids')] + p['pos'][integer('pos')] + p['seg'][integer('seg')]
    mask = torch.tensor(batch['mask_bias'], dtype=h.dtype)
    for i in range(layers):
        prefix = f'layers.{i}.'
        def q(name): return p[prefix+name]
        normalized = F.layer_norm(h, (d,), q('ln1_g'), q('ln1_b'), 1e-5)
        b, t, _ = h.shape
        projections = normalized @ q('w_qkv') + q('b_qkv')
        projections = projections.reshape(b, t, 3, heads, d//heads).permute(2,0,3,1,4)
        query, key, value = projections.unbind(0)
        attention = torch.softmax((query * (d//heads)**-.5) @ key.transpose(-1,-2) + mask, dim=-1)
        attended = (attention @ value).permute(0,2,1,3).reshape(b,t,d)
        h = h + (attended @ q('w_o') + q('b_o'))
        normalized = F.layer_norm(h, (d,), q('ln2_g'), q('ln2_b'), 1e-5)
        hidden = F.gelu(normalized @ q('w_1') + q('b_1'), approximate='tanh')
        h = h + (hidden @ q('w_2') + q('b_2'))
    h = F.layer_norm(h, (d,), p['lnf_g'], p['lnf_b'], 1e-5).reshape(-1,d)
    scores = (h[integer('opt_rows')] @ p['w_opt']).reshape(-1)
    binary = (h[integer('noul_rows')] @ p['w_noul'] + p['b_noul']).reshape(-1)
    return scores, binary


def loss(parameters, batch, d, heads, layers):
    scores, binary = forward(parameters, batch, d, heads, layers)
    # All targets are exact sampled 0/1 labels; conversion preserves values.
    categoricals = []
    for group in range(batch['n_groups']):
        mask = batch['opt_group'] == group
        label = int(batch['opt_target'][mask].argmax())
        categoricals.append(F.cross_entropy(scores[mask].reshape(1,-1), torch.tensor([label])))
    categorical_loss = torch.stack(categoricals).mean() if categoricals else scores.sum()*0.
    targets = torch.tensor(batch['noul_target'], dtype=binary.dtype)
    binary_loss = F.binary_cross_entropy_with_logits(binary, targets) if binary.numel() else binary.sum()*0.
    return categorical_loss + binary_loss, (scores,binary)


def difference(first, second, atol, rtol):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    delta = a-b
    maximum = float(np.abs(delta).max()) if delta.size else 0.
    scale = max(float(np.abs(a).max()), float(np.abs(b).max())) if delta.size else 0.
    norm_scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    return {'max_abs':maximum, 'relative_l2':float(np.linalg.norm(delta)/norm_scale),
            'reference_max_abs':float(np.abs(b).max()) if b.size else 0.,
            'comparison_pass':maximum <= atol+rtol*scale}


def gradients(model, parameters, atol, rtol):
    comparisons = {}
    total_squared, total_reference_squared = 0., 0.
    for name,param in named_parameters(model).items():
        torch_param = parameters[name]
        first = np.zeros_like(param.data) if param.grad is None else param.grad
        second = np.zeros_like(param.data) if torch_param.grad is None else torch_param.grad.detach().numpy()
        row = difference(first, second, atol, rtol)
        row.update(parameter_dtype=str(param.data.dtype), numpy_gradient_dtype=None if param.grad is None else str(param.grad.dtype),
                   torch_gradient_dtype=None if torch_param.grad is None else str(torch_param.grad.dtype),
                   numpy_gradient_max_abs=float(np.abs(first).max()))
        if name.endswith('.b_qkv'):
            d=model.d
            row['key_bias'] = difference(first[d:2*d],second[d:2*d],atol,rtol)
            row['key_bias']['numpy_max_abs'] = float(np.abs(first[d:2*d]).max())
        comparisons[name]=row
        total_squared += float(((first.astype(float)-second.astype(float))**2).sum())
        total_reference_squared += float((second.astype(float)**2).sum())
    return {'all_pass':all(r['comparison_pass'] for r in comparisons.values()),
            'global_relative_l2':float(np.sqrt(total_squared/max(total_reference_squared,1e-60))),
            'maximum_absolute_error':max(r['max_abs'] for r in comparisons.values()),
            'parameters':comparisons}


def make_model(architecture, dtype, base=None):
    ag.set_dtype(dtype)
    model=TinyS1(len(D.VOCAB), **architecture, seed=0)
    if base is not None:
        for name,p in named_parameters(model).items(): p.data[...] = base[name].astype(dtype)
    return model


def source_hashes():
    return {name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
            for name in ('autograd.py','model.py','data.py','train.py')}


def batches(seed, count, batch_size, target_dtype):
    rng=D.make_rng(seed)
    result=[]
    for _ in range(count):
        batch=D.collate([D.sample_request(rng) for _ in range(batch_size)])
        if target_dtype is not None:
            batch['opt_target']=batch['opt_target'].astype(target_dtype)
            batch['noul_target']=batch['noul_target'].astype(target_dtype)
        result.append(batch)
    return result


def data_hash(batches):
    h=hashlib.sha256()
    for batch in batches:
        for name in ('ids','pos','seg','mask_bias','opt_rows','opt_group','opt_target','noul_rows','noul_target'):
            value=np.ascontiguousarray(batch[name])
            h.update((name+str(value.shape)+str(value.dtype)).encode());h.update(value.tobytes())
    return h.hexdigest()


def torch_optimizer(model, parameters):
    decay={id(p) for p in model.matrices()}
    names=named_parameters(model)
    groups=[{'params':[parameters[n] for n,p in names.items() if id(p) in decay], 'weight_decay':.01},
            {'params':[parameters[n] for n,p in names.items() if id(p) not in decay], 'weight_decay':0.}]
    return torch.optim.AdamW(groups,lr=.003,betas=(.9,.98),eps=1e-8,foreach=False,fused=False)


def run_model_case(architecture,dtype,expression,targets,steps,batch_size,checkpoint=None):
    # FP64 begins at exactly the same FP32-representable parameter values.
    initial=make_model(architecture,np.float32)
    base={n:p.data.copy() for n,p in named_parameters(initial).items()}
    if checkpoint is not None:
        with np.load(checkpoint,allow_pickle=False) as arrays:
            keys=sorted(arrays.files,key=lambda name:int(name.split('_')[1]))
            assert len(keys)==len(base)
            for name,key in zip(base,keys):
                assert arrays[key].shape==base[name].shape
                base[name]=arrays[key].copy()
    model=make_model(architecture,dtype,base)
    torch_dtype=torch.float64 if dtype==np.float64 else torch.float32
    parameters=torch_parameters(model,torch_dtype)
    opt=ag.AdamW(model.parameters(),decay=model.matrices())
    reference_optimizer=torch_optimizer(model,parameters)
    original_sigmoid=ag.sigmoid_np
    if expression=='direct':ag.sigmoid_np=lambda z:1./(1.+np.exp(-z))
    atol,rtol=(1e-10,1e-9) if dtype==np.float64 else (3e-6,3e-5)
    audit_seed=1000 if checkpoint is None else 91001
    stream=batches(audit_seed,steps,batch_size,None if targets=='canonical_float64' else dtype)
    report={'dtype':str(np.dtype(dtype)),'sigmoid':expression,'target_mode':targets,
            'weight_state':'initial' if checkpoint is None else 'explicit_checkpoint',
            'checkpoint_name':None if checkpoint is None else checkpoint.name,
            'checkpoint_sha256':None if checkpoint is None else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'audit_batch_seed':audit_seed,
            'target_dtypes':{n:str(stream[0][n].dtype) for n in ('opt_target','noul_target')},
            'atol':atol,'rtol':rtol,'data_sha256':data_hash(stream),'steps':[]}
    try:
        for step,batch in enumerate(stream):
            opt.zero_grad();reference_optimizer.zero_grad(set_to_none=True)
            np_loss,np_logits=model.loss(batch)
            torch_loss,torch_logits=loss(parameters,batch,model.d,model.h,len(model.layers))
            np_loss.backward();torch_loss.backward()
            row={'step':step,'numpy_loss':float(np_loss.data),'torch_loss':float(torch_loss.detach()),
                 'loss_abs_error':abs(float(np_loss.data)-float(torch_loss.detach())),
                 'logits':{name:difference(n.data,t.detach().numpy(),atol,rtol)
                           for name,n,t in zip(('categorical','binary'),np_logits,torch_logits)},
                 'gradients':gradients(model,parameters,atol,rtol)}
            opt.lr=lr_at(step,4000)
            reference_optimizer.param_groups[0]['lr']=opt.lr
            reference_optimizer.param_groups[1]['lr']=opt.lr
            opt.step();reference_optimizer.step()
            row['parameters_after_step']={name:difference(p.data,parameters[name].detach().numpy(),atol,rtol)
                                          for name,p in named_parameters(model).items()}
            report['steps'].append(row)
    finally:ag.sigmoid_np=original_sigmoid
    report['first_step_gradients_pass']=report['steps'][0]['gradients']['all_pass']
    report['first_step_loss_pass']=report['steps'][0]['loss_abs_error'] <= atol+rtol*abs(report['steps'][0]['torch_loss'])
    report['max_final_parameter_difference']=max(r['max_abs'] for r in report['steps'][-1]['parameters_after_step'].values())
    return report


def optimizer_only(architecture,dtype,steps=5):
    model=make_model(architecture,dtype)
    parameters=torch_parameters(model,torch.float64 if dtype==np.float64 else torch.float32)
    opt=ag.AdamW(model.parameters(),lr=.003,decay=model.matrices())
    reference=torch_optimizer(model,parameters)
    rng=np.random.default_rng(71351)
    reports=[]
    atol,rtol=(1e-12,1e-10) if dtype==np.float64 else (2e-7,2e-5)
    for step in range(steps):
        for i,(name,param) in enumerate(named_parameters(model).items()):
            scale=(.01,1e-8,0.)[i%3]
            grad=rng.normal(size=param.shape).astype(dtype)*scale
            param.grad=grad.copy()
            parameters[name].grad=torch.tensor(grad.copy(),dtype=parameters[name].dtype)
        opt.step();reference.step()
        row={'step':step,'parameters':{},'moments':{}}
        for i,(name,param) in enumerate(named_parameters(model).items()):
            state=reference.state[parameters[name]]
            row['parameters'][name]=difference(param.data,parameters[name].detach().numpy(),atol,rtol)
            row['moments'][name]={kind:difference(value,state[key].detach().numpy(),atol,rtol)
                                  for kind,value,key in (('first',opt.m[i],'exp_avg'),('second',opt.v[i],'exp_avg_sq'))}
        reports.append(row)
    return {'dtype':str(np.dtype(dtype)),'atol':atol,'rtol':rtol,'steps':reports,
            'all_pass':all(r['comparison_pass'] for row in reports for r in row['parameters'].values())
                       and all(r['comparison_pass'] for row in reports for values in row['moments'].values() for r in values.values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--stable-checkpoint',type=Path,help='Optional trained stable-expression NPZ checkpoint')
    parser.add_argument('--direct-checkpoint',type=Path,help='Optional trained direct-expression NPZ checkpoint')
    args=parser.parse_args()
    if args.steps<1 or args.batch<1:
        parser.error('steps and batch must be positive')
    checkpoints=[(label,path.resolve()) for label,path in
                 (('stable',args.stable_checkpoint),('direct',args.direct_checkpoint)) if path is not None]
    for label,path in checkpoints:
        if not path.is_file():parser.error(f'{label} checkpoint does not exist: {path}')
    result={'versions':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,
                        'platform':platform.platform(),'torch_threads':torch.get_num_threads(),
                        'numpy_blas_threads_environment':os.environ.get('OPENBLAS_NUM_THREADS')},
            'source_sha256':source_hashes(),
            'reference_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Independent equations in Torch; FP64 algebra check, FP32 descriptive comparison. No cross-framework bitwise equivalence asserted.',
            'trained_state_coverage':{'status':'explicit_checkpoints' if checkpoints else 'not_requested',
                                      'provided_labels':[label for label,_ in checkpoints],
                                      'note':'Passing initial/optimizer checks does not imply trained-state checks when checkpoints are omitted.'},
            'architectures':{}}
    for label,architecture in (('small',{'d':8,'n_heads':2,'n_layers':2,'d_ff':16}),
                               ('default',{'d':64,'n_heads':4,'n_layers':2,'d_ff':256})):
        row={'architecture':architecture,'model_checks':[],'optimizer_only':[], 'trained_checkpoint_checks':[]}
        for dtype in (np.float64,np.float32):
            row['optimizer_only'].append(optimizer_only(architecture,dtype))
            for expression in ('stable','direct'):
                for target_mode in ('canonical_float64','parameter_dtype'):
                    check=run_model_case(architecture,dtype,expression,target_mode,args.steps,args.batch)
                    row['model_checks'].append(check)
                    print(json.dumps({'architecture':label,'dtype':check['dtype'],'sigmoid':expression,'targets':target_mode,
                                      'initial_gradient_pass':check['first_step_gradients_pass'],
                                      'initial_gradient_max_abs':check['steps'][0]['gradients']['maximum_absolute_error'],
                                      'final_parameter_max_abs':check['max_final_parameter_difference']}),flush=True)
        if label=='default':
            for variant,checkpoint in checkpoints:
                for dtype in (np.float64,np.float32):
                    for expression in ('stable','direct'):
                        for target_mode in ('canonical_float64','parameter_dtype'):
                            check=run_model_case(architecture,dtype,expression,target_mode,1,args.batch,checkpoint)
                            check['provided_checkpoint_label']=variant
                            row['trained_checkpoint_checks'].append(check)
                            print(json.dumps({'architecture':label,'weight_state':variant,'dtype':check['dtype'],
                                              'sigmoid':expression,'targets':target_mode,
                                              'initial_gradient_pass':check['first_step_gradients_pass'],
                                              'initial_gradient_max_abs':check['steps'][0]['gradients']['maximum_absolute_error']}),flush=True)
        result['architectures'][label]=row
    result['summary']={
        'fp64_initial_gradient_checks_pass':all(c['first_step_gradients_pass'] and c['first_step_loss_pass']
                              for a in result['architectures'].values() for c in a['model_checks']+a['trained_checkpoint_checks'] if c['dtype']=='float64'),
        'fp32_initial_gradient_checks_pass':all(c['first_step_gradients_pass'] and c['first_step_loss_pass']
                              for a in result['architectures'].values() for c in a['model_checks']+a['trained_checkpoint_checks'] if c['dtype']=='float32'),
        'controlled_optimizer_checks_pass':all(c['all_pass'] for a in result['architectures'].values() for c in a['optimizer_only'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['summary']),flush=True)
    if not all(result['summary'].values()):
        raise SystemExit(2)


if __name__=='__main__':main()
