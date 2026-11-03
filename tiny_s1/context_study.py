"""Matched Choice-only experiments with isolated or interacting option blocks."""
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import autograd as ag
import data as D
from model import TinyS1
from train import lr_at
from baselines import metrics

MODES = ('isolated', 'count', 'random_count', 'set')
WORLDS = ('fixed_q', 'fixed_odds')


def choice_request(rng, world, topics=None, n_opts=(2, 6)):
    if world not in WORLDS:
        raise ValueError(world)
    probability = None if world == 'fixed_q' else lambda n: 7.0 / (7.0 + n - 1)
    req = D.sample_request(rng, topics=topics, n_opts=n_opts, q_topic=probability)
    req['questions'] = req['questions'][:1]
    return req


def make_requests(world, seed, n, heldout=False):
    rng = D.make_rng(seed)
    return [choice_request(rng, world, topics=D.HELD_OUT if heldout else None,
                           n_opts=(2, 4) if heldout else (2, 6)) for _ in range(n)]


def make_batch(requests, mode, random_context=None):
    if mode not in MODES:
        raise ValueError(mode)
    batch = D.collate(requests)
    context = np.zeros_like(batch['ids'])
    cursor = 0
    if mode == 'random_count' and (random_context is None or
                                  len(random_context) != sum(len(r['questions']) for r in requests)):
        raise ValueError('Supply one independent context value per question')
    for i, req in enumerate(requests):
        for qi, q in enumerate(req['questions']):
            value = len(q.get('options', [])) if mode == 'count' else (
                int(random_context[cursor]) if mode == 'random_count' else 0)
            if not 0 <= value <= 6:
                raise ValueError('This experiment supports context counts from 0 to 6')
            context[i, (batch['kind'][i] == D.QUESTION) & (batch['qid'][i] == qi)] = value
            cursor += 1
    batch['context_ids'] = context
    if mode == 'set':
        kind, qid = batch['kind'], batch['qid']
        ki, kj = kind[:, :, None], kind[:, None, :]
        own_question = qid[:, :, None] == qid[:, None, :]
        # Pool the set through the question. Option markers keep their own
        # local block; giving identical markers identical key sets collapses
        # their scores and destroys candidate identity.
        extra = (ki == D.QUESTION) & (kj == D.OPTION) & own_question
        allowed = (batch['mask_bias'][:, 0] == 0) | extra
        batch['mask_bias'] = np.where(allowed, 0., -1e9)[:, None]
    return batch


class ContextModel(TinyS1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = ag.Tensor(np.zeros((7, self.d)), requires_grad=True)

    def parameters(self):
        return super().parameters() + [self.context]

    def encode(self, batch):
        x = (ag.take_rows(self.tok, batch['ids']) + ag.take_rows(self.pos, batch['pos'])
             + ag.take_rows(self.seg, batch['seg']))
        active = (batch['context_ids'] > 0)[..., None]
        x = x + ag.take_rows(self.context, batch['context_ids']) * ag.Tensor(active)
        bias = ag.Tensor(batch['mask_bias'])
        for layer in self.layers:
            x = x + self.attention(ag.layer_norm(x, layer['ln1_g'], layer['ln1_b']), layer, bias)
            h = ag.gelu(ag.layer_norm(x, layer['ln2_g'], layer['ln2_b']) @ layer['w_1'] + layer['b_1'])
            x = x + (h @ layer['w_2'] + layer['b_2'])
        x = ag.layer_norm(x, self.lnf_g, self.lnf_b)
        b, t, d = x.shape
        return x.reshape(b * t, d)


def parameter_hash(model):
    digest = hashlib.sha256()
    for param in model.parameters():
        digest.update(str(param.shape).encode())
        digest.update(param.data.tobytes())
    return digest.hexdigest()


def update_data_hash(digest, batch):
    for name in ('ids', 'pos', 'seg', 'opt_rows', 'opt_group', 'opt_target'):
        array = np.ascontiguousarray(batch[name])
        digest.update((name + str(array.shape) + str(array.dtype)).encode())
        digest.update(array.tobytes())


def predictions(model, requests, mode, context_seed, batch_size=32):
    context = np.random.default_rng(context_seed).integers(2, 7, size=len(requests))
    n = np.array([len(r['questions'][0]['options']) for r in requests])
    scores = np.full((len(requests), int(n.max())), -np.inf)
    bayes = np.zeros_like(scores)
    targets = np.zeros(len(requests), int)
    for start in range(0, len(requests), batch_size):
        items = requests[start:start + batch_size]
        batch = make_batch(items, mode, context[start:start + len(items)])
        s, _ = model.forward(batch)
        for j, (_, ref, question) in enumerate(batch['meta']):
            a, b, _ = ref
            scores[start + j, :b-a] = s.data[a:b].astype(float)
            bayes[start + j, :b-a] = question['bayes']
            targets[start + j] = int(np.argmax(question['target']))
    if not np.all(np.isfinite(scores[np.arange(scores.shape[1])[None] < n[:, None]])):
        raise FloatingPointError('Non-finite prediction scores')
    return {'scores': scores, 'n': n, 'targets': targets, 'bayes': bayes, 'random_context': context}


def probabilities(scores, temperature=1.):
    scaled = scores / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled - maximum)
    normalizer = exponent.sum(axis=1, keepdims=True)
    return exponent / normalizer, scaled - maximum - np.log(normalizer)


def score_metrics(records, temperature=1.):
    p, log_p = probabilities(records['scores'], temperature)
    bayes, targets = records['bayes'], records['targets']
    out = metrics(p, log_p, targets, bayes)
    entropy = (bayes * np.log(np.clip(bayes, 1e-300, 1))).sum(axis=1)
    cross_entropy = -(bayes * np.where(bayes > 0, log_p, 0.)).sum(axis=1)
    out['posterior_kl'] = float(np.maximum(0., cross_entropy + entropy).mean())
    out['posterior_squared_error'] = float(((p-bayes)**2).sum(axis=1).mean())
    return out


def temperature_on_validation(records):
    grid = np.exp(np.linspace(np.log(.05), np.log(200.), 400))
    index = np.arange(len(records['targets']))
    losses = [-probabilities(records['scores'], t)[1][index, records['targets']].mean() for t in grid]
    return float(grid[int(np.argmin(losses))])


def option_order_error(model, requests, mode, context_seed):
    modified = copy.deepcopy(requests)
    rng = np.random.default_rng(91817)
    permutations = []
    for req in modified:
        q = req['questions'][0]
        perm = rng.permutation(len(q['options']))
        for key in ('options', 'option_ids', 'target'):
            q[key] = [q[key][i] for i in perm]
        q['bayes'] = q['bayes'][perm]
        permutations.append(perm)
    first = probabilities(predictions(model, requests, mode, context_seed)['scores'])[0]
    second = probabilities(predictions(model, modified, mode, context_seed)['scores'])[0]
    return float(max(np.abs(first[i, :len(perm)][perm] - second[i, :len(perm)]).max()
                     for i, perm in enumerate(permutations)))


def atomic_json(path, data):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def run(args):
    ag.set_dtype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'results.json').exists():
        raise FileExistsError('Completed results exist; use a new output directory')
    model = ContextModel(len(D.VOCAB), seed=args.seed)
    params = model.parameters()
    optimizer = ag.AdamW(params, decay=model.matrices())
    initial = parameter_hash(model)
    rng = D.make_rng(1000 + args.seed)
    context_rng = np.random.default_rng(2000 + args.seed)
    data_hash = hashlib.sha256()
    start, ema, log = time.perf_counter(), None, []
    config = {key: getattr(args, key) for key in ('world', 'mode', 'seed', 'steps', 'batch')}
    config.update({'task': 'Choice only; state retains urgency and flag distractors',
                   'source_commit': os.environ.get('EXPERIMENT_SOURCE_COMMIT'),
                   'parameters': sum(p.data.size for p in params), 'context_parameters': model.context.data.size,
                   'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(),
                   'processor': platform.processor(), 'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
                   'validation_seed': 72001, 'test_seed': 72002, 'heldout_seed': 72003,
                   'initial_parameters_sha256': initial, 'source_sha256': {}})
    for name in ('context_study.py', 'data.py', 'autograd.py', 'model.py', 'train.py', 'baselines.py'):
        config['source_sha256'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    atomic_json(args.output_dir / 'config.json', config)
    context_counts = np.zeros((7, 7), dtype=np.int64)
    for step in range(args.steps):
        requests = [choice_request(rng, args.world) for _ in range(args.batch)]
        batch = make_batch(requests, args.mode, context_rng.integers(2, 7, size=args.batch))
        for i, req in enumerate(requests):
            n = len(req['questions'][0]['options'])
            value = batch['context_ids'][i, len(req['state'])+1]
            context_counts[n, value] += 1
        update_data_hash(data_hash, batch)
        optimizer.zero_grad()
        optimizer.lr = lr_at(step, args.steps)
        loss, _ = model.loss(batch)
        if not np.isfinite(loss.data):
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        optimizer.step()
        ema = float(loss.data) if ema is None else .98 * ema + .02 * float(loss.data)
        if step % 100 == 0 or step == args.steps-1:
            entry = {'step': step, 'loss_ema': ema, 'seconds': time.perf_counter()-start}
            log.append(entry)
            print(json.dumps(entry), flush=True)
            atomic_json(args.output_dir / 'training-progress.json', entry)
    training_seconds = time.perf_counter()-start
    checkpoint = args.output_dir / 'model.npz'
    temp = args.output_dir / 'model.tmp.npz'
    np.savez(temp, *[p.data for p in params])
    temp.replace(checkpoint)
    result = {'config': config, 'training_seconds': training_seconds, 'training_log': log,
              'option_count_by_context_index': context_counts.tolist(),
              'training_data_sha256': data_hash.hexdigest(),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'pilot': args.skip_evaluation}
    if not args.skip_evaluation:
        validation = predictions(model, make_requests(args.world, 72001, 2000), args.mode, 73001)
        test_requests = make_requests(args.world, 72002, 4000)
        test = predictions(model, test_requests, args.mode, 73002)
        heldout = predictions(model, make_requests(args.world, 72003, 2000, heldout=True), args.mode, 73003)
        temperature = temperature_on_validation(validation)
        result.update({'test': score_metrics(test), 'heldout_vocabulary': score_metrics(heldout),
                       'temperature': temperature, 'test_temperature_scaled': score_metrics(test, temperature),
                       'by_option_count': {}, 'option_order_max_abs_difference':
                           option_order_error(model, test_requests[:64], args.mode, 73002)})
        for n in range(2, 7):
            mask = test['n'] == n
            subset = {key: values[mask] for key, values in test.items()}
            result['by_option_count'][str(n)] = score_metrics(subset)
        if result['option_order_max_abs_difference'] > 1e-4:
            raise AssertionError('Option-order invariance check failed')
        np.savez_compressed(args.output_dir / 'predictions.npz',
                            **{'test_'+k: v for k,v in test.items()},
                            **{'heldout_'+k: v for k,v in heldout.items()})
    result['total_seconds'] = time.perf_counter()-start
    atomic_json(args.output_dir / 'results.json', result)
    print(json.dumps({'complete': True, 'mode': args.mode, 'world': args.world,
                      'seed': args.seed, 'seconds': result['total_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', choices=WORLDS, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--skip-evaluation', action='store_true')
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.batch < 1:
        parser.error('steps and batch must be positive')
    run(arguments)
