"""train.py: train TinyS1 with the log score (a strictly proper scoring rule).

The model never sees a probability. It sees sampled outcomes (which team, which
urgency level, true/false) and is penalised by -log p(outcome). Because the log
score is strictly proper, its expected value is minimised only by the true
conditional probabilities; that is the whole calibration story at training time.

Usage:
  python train.py --run stream          # fresh synthetic requests every step
  python train.py --run fixed           # a small fixed dataset, many epochs
  python train.py --run other           # stream, sometimes offering "other"
"""
import argparse, json, time
import numpy as np
import autograd as ag
import data as D
from model import TinyS1


def sampler(run, seed):
    rng = D.make_rng(seed)
    if run == "fixed":
        pool = [D.sample_request(rng) for _ in range(512)]
        while True:
            yield rng.choice(pool)
    while True:
        if run == "other":
            u = rng.random()
            if u < 0.15:
                yield D.sample_request(rng, oos=True, offer_other=True)
                continue
            if u < 0.30:
                yield D.sample_request(rng, offer_other=True)
                continue
        yield D.sample_request(rng)


def lr_at(step, total, peak=3e-3, warmup=200):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["stream", "fixed", "other"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = TinyS1(len(D.VOCAB), seed=args.seed)
    params = model.parameters()
    opt = ag.AdamW(params, decay=model.matrices())
    gen = sampler(args.run, seed=1000 + args.seed)
    log, t0, ema = [], time.time(), None
    for step in range(args.steps):
        batch = D.collate([next(gen) for _ in range(args.batch)])
        opt.lr = lr_at(step, args.steps)
        opt.zero_grad()
        loss, _ = model.loss(batch)
        loss.backward()
        opt.step()
        ema = float(loss.data) if ema is None else 0.98 * ema + 0.02 * float(loss.data)
        if step % 100 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss_ema": round(ema, 4), "sec": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    np.savez(f"ckpt_{args.run}.npz", *[p.data for p in params])
    json.dump({"run": args.run, "steps": args.steps, "batch": args.batch, "seed": args.seed,
               "train_seconds": round(time.time() - t0, 1), "log": log},
              open(f"trainlog_{args.run}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
