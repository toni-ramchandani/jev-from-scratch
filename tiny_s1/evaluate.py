"""evaluate.py: measure what the contract does NOT guarantee.

Everything here is a measurement of OUR toy model on OUR synthetic data. None
of it says anything about Jev's accuracy or calibration.
Writes results.json and figures/*.png.
"""
import json, time, copy, argparse, sys, platform, os, hashlib
from pathlib import Path
import numpy as np
import autograd as ag
import data as D
from model import TinyS1, confidence

RUNS = ["stream", "fixed", "other"]
BINS = 10


def load(run, checkpoint_dir=Path(".")):
    m = TinyS1(len(D.VOCAB))
    arrs = np.load(Path(checkpoint_dir) / f"ckpt_{run}.npz", allow_pickle=False)
    if len(arrs.files) != len(m.parameters()):
        raise ValueError("Checkpoint parameter count does not match model")
    for p, k in zip(m.parameters(), sorted(arrs.files, key=lambda s: int(s.split("_")[1]))):
        if p.shape != arrs[k].shape:
            raise ValueError(f"Checkpoint shape mismatch: {k}")
        p.data = arrs[k].astype(np.float32)
    return m


def make_set(seed, n, **kw):
    r = D.make_rng(seed)
    return [D.sample_request(r, **kw) for _ in range(n)]


def run_model(model, reqs, bs=32):
    """Return per-question records: raw option scores / noul logit + targets + bayes."""
    recs = []
    for i in range(0, len(reqs), bs):
        b = D.collate(reqs[i:i + bs])
        s, z = model.forward(b)
        s, z = s.data.astype(float), z.data.astype(float)
        for typ, ref, q in b["meta"]:
            if typ == "noul":
                recs.append(dict(type="noul", logit=z[ref], y=q["target"], bayes=q["bayes"]))
            else:
                a0, a1, _ = ref
                recs.append(dict(type=typ, scores=s[a0:a1], y=int(np.argmax(q["target"])),
                                 bayes=q["bayes"], q=q))
    return recs


def softmax(v, T=1.0):
    e = np.exp((v - v.max()) / T)
    return e / e.sum()


def log_loss(scores, target, T=1.0):
    v = np.asarray(scores, float) / T
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()) - v[target])


def wilson_interval(correct, z=1.959963984540054):
    n = len(correct)
    p = float(np.mean(correct))
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def reliability(conf, correct):
    edges = np.linspace(0, 1, BINS + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), conf=float(conf[m].mean()), acc=float(correct[m].mean())))
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece), rows


def option_metrics(recs, typ, T=1.0):
    rs = [r for r in recs if r["type"] == typ]
    P = [softmax(r["scores"], T) for r in rs]
    y = np.array([r["y"] for r in rs])
    top = np.array([p.max() for p in P]); pred = np.array([p.argmax() for p in P])
    correct = (pred == y).astype(float)
    nll = np.mean([log_loss(r["scores"], t, T) for r, t in zip(rs, y)])
    brier = np.mean([((p - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(P, y)])
    ece, rel = reliability(top, correct)
    out = dict(n=len(rs), acc=correct.mean(), acc_wilson_95=wilson_interval(correct),
               nll=nll, brier=brier, ece=ece, reliability=rel,
               mean_top_prob=top.mean(), mean_confidence=float(np.mean([confidence(p) for p in P])))
    if rs and rs[0]["bayes"] is not None and len(rs[0]["bayes"]) == len(rs[0]["scores"]):
        B = [r["bayes"] for r in rs]
        out["tv_to_bayes"] = float(np.mean([0.5 * np.abs(p - b).sum() for p, b in zip(P, B)]))
        btop = np.array([b.max() for b in B]); bpred = np.array([b.argmax() for b in B])
        bece, brel = reliability(btop, (bpred == y).astype(float))
        out["bayes"] = dict(acc=float((bpred == y).mean()),
                            nll=float(-np.mean([np.log(b[t]) for b, t in zip(B, y)])),
                            brier=float(np.mean([((b - np.eye(len(b))[t]) ** 2).sum() for b, t in zip(B, y)])),
                            ece=bece, reliability=brel)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def noul_metrics(recs, T=1.0):
    rs = [r for r in recs if r["type"] == "noul"]
    logits = np.array([r["logit"] / T for r in rs])
    p = ag.sigmoid_np(logits); y = np.array([r["y"] for r in rs])
    p = np.clip(p, 1e-12, 1 - 1e-12)  # a memorising model can output exact 0 or 1
    b = np.array([r["bayes"] for r in rs])
    ece, rel = reliability(p, y)  # binary: is p(true) itself honest?
    bece, _ = reliability(b, y)
    return dict(n=len(rs), acc=float(((p > 0.5) == y).mean()),
                nll=float(np.mean(np.logaddexp(0, logits) - y * logits)),
                brier=float(np.mean((p - y) ** 2)), ece=ece, mean_abs_to_bayes=float(np.abs(p - b).mean()),
                bayes=dict(acc=float(((b > 0.5) == y).mean()), brier=float(np.mean((b - y) ** 2)), ece=bece),
                reliability=rel)


def fit_temperature(recs, kind):
    grid = np.exp(np.linspace(np.log(0.05), np.log(200), 400))  # wide: a memoriser wants a huge T
    if kind == "option":
        rs = [r for r in recs if r["type"] == "choice"]
        f = lambda T: np.mean([log_loss(r["scores"], r["y"], T) for r in rs])
    else:
        rs = [r for r in recs if r["type"] == "noul"]
        z = np.array([r["logit"] for r in rs]); y = np.array([r["y"] for r in rs])
        def f(T):
            return np.mean(np.logaddexp(0, z / T) - y * z / T)
    losses = [f(T) for T in grid]
    return float(grid[int(np.argmin(losses))])


def forced_choice(recs, other_name=None):
    rs = [r for r in recs if r["type"] == "choice"]
    P = [softmax(r["scores"]) for r in rs]
    top = np.array([p.max() for p in P])
    out = dict(n=len(rs), mean_top_prob=float(top.mean()),
               frac_top_ge_0_8=float((top >= 0.8).mean()), frac_top_ge_0_9=float((top >= 0.9).mean()))
    if other_name:
        idx = [[o[0] for o in r["q"]["options"]].index(other_name) for r in rs]
        po = np.array([p[i] for p, i in zip(P, idx)])
        picks_other = np.array([p.argmax() == i for p, i in zip(P, idx)])
        out.update(mean_p_other=float(po.mean()), frac_picks_other=float(picks_other.mean()))
    return out


def order_invariance(model, reqs):
    shuffled = copy.deepcopy(reqs)
    r = np.random.default_rng(3)
    perms = []
    for req in shuffled:
        q = req["questions"][0]
        perm = r.permutation(len(q["options"]))
        q["options"] = [q["options"][i] for i in perm]; q["target"] = [q["target"][i] for i in perm]
        if "option_ids" in q:
            q["option_ids"] = [q["option_ids"][i] for i in perm]
        q["bayes"] = q["bayes"][perm]
        perms.append(perm)
    a = [softmax(x["scores"]) for x in run_model(model, reqs) if x["type"] == "choice"]
    b = [softmax(x["scores"]) for x in run_model(model, shuffled) if x["type"] == "choice"]
    return float(max(np.abs(p[perm] - q).max() for p, q, perm in zip(a, b, perms)))


def timing(model, reqs, bs=64, reps=3):
    """Packed (all questions in one sequence) vs one sequence per question."""
    split = [{"state": r["state"], "questions": [q]} for r in reqs for q in r["questions"]]
    def clock(items):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            for i in range(0, len(items), bs):
                model.forward(D.collate(items[i:i + bs]))
            best = min(best, time.perf_counter() - t)
        return best
    def tokens(items):
        return int(sum(len(D.encode_request(x)["ids"]) for x in items))
    tp, ts = clock(reqs), clock(split)
    return dict(requests=len(reqs), questions=len(split),
                packed=dict(sequences=len(reqs), tokens=tokens(reqs), ms_per_request=1000 * tp / len(reqs)),
                separate=dict(sequences=len(split), tokens=tokens(split), ms_per_request=1000 * ts / len(reqs)))


def example_answer(model):
    req = make_set(4242, 1, name_mode="topic")[0]
    out = model.answer(D.collate([req]))
    return dict(state=" ".join(req["state"]),
                questions=[{"type": q["type"], "instructions": " ".join(q["instr"]),
                            "criteria": [o[0] + ": " + " ".join(o[1]) for o in q.get("options", [])]} for q in req["questions"]],
                answers=out,
                bayes=dict(choice=np.round(req["questions"][0]["bayes"], 3).tolist(),
                           score=np.round(req["questions"][1]["bayes"], 3).tolist(),
                           noul=[round(q["bayes"], 3) for q in req["questions"][2:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("../artifacts/supplied"))
    ap.add_argument("--output-dir", type=Path, default=Path("../results/reproduced"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sets = dict(
        val=make_set(5001, 2000),
        test=make_set(5002, 4000),
        hidden=make_set(5003, 2000, name_mode="hidden"),
        oos=make_set(5005, 2000, oos=True),
        oos_other=make_set(5006, 2000, oos=True, offer_other=True),
        in_other=make_set(5007, 2000, offer_other=True),
        heldout=make_set(5008, 2000, topics=D.HELD_OUT, n_opts=(2, 4)),
    )
    res = {"config": dict(bins=BINS, test_requests=4000, numpy=np.__version__,
                         python=sys.version, platform=platform.platform(), inference_batch=32,
                         openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                         seeds=dict(val=5001, test=5002, hidden=5003, oos=5005,
                                    oos_other=5006, in_other=5007, heldout=5008),
                         note="Toy synthetic experiment; not an evaluation of Jev")}
    for run in RUNS:
        model = load(run, args.checkpoint_dir)
        R = {k: run_model(model, v) for k, v in sets.items()}
        T_opt, T_noul = fit_temperature(R["val"], "option"), fit_temperature(R["val"], "noul")
        out = dict(
            checkpoint_sha256=hashlib.sha256((args.checkpoint_dir / f"ckpt_{run}.npz").read_bytes()).hexdigest(),
            T_option=T_opt, T_noul=T_noul,
            choice=option_metrics(R["test"], "choice"),
            choice_T=option_metrics(R["test"], "choice", T_opt),
            score=option_metrics(R["test"], "score"),
            noul=noul_metrics(R["test"]),
            noul_T=noul_metrics(R["test"], T_noul),
            hidden_names=option_metrics(R["hidden"], "choice"),
            heldout_topics=option_metrics(R["heldout"], "choice"),
            forced_choice_in=forced_choice(R["test"]),
            forced_choice_oos=forced_choice(R["oos"]),
            oos_with_other=forced_choice(R["oos_other"], "other"),
            in_with_other=forced_choice(R["in_other"], "other") | {"acc": option_metrics(R["in_other"], "choice")["acc"]},
            order_invariance_max_abs_dp=order_invariance(model, sets["test"][:500]),
        )
        hs = [r for r in R["heldout"] if r["type"] == "choice"]
        out["heldout_topics"]["chance_acc"] = float(np.mean([1 / len(r["scores"]) for r in hs]))
        if run == "stream":
            out["timing"] = timing(model, sets["test"][:512])
            out["example"] = example_answer(model)
        res[run] = out
        print(run, json.dumps({k: (round(v["acc"], 3), round(v.get("ece", 0), 3)) for k, v in out.items()
                               if isinstance(v, dict) and "acc" in v}))
    for run in RUNS:
        res[run]["train"] = json.load(open(args.checkpoint_dir / f"trainlog_{run}.json"))
        res[run]["train"].pop("log")
    json.dump(res, open(args.output_dir / "results.json", "w"), indent=1, default=float)
    if not args.no_figures:
        figures(res, sets, args.output_dir / "figures")


# ---- figures ---------------------------------------------------------------
def figures(res, sets, output_dir=Path("figures")):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    ink, muted, grid = "#1f2328", "#6e7781", "#d0d7de"
    c1, c2, c3 = "#0969da", "#cf222e", "#1a7f37"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})

    # Reliability diagram from the recorded calibration bins.
    dark, light = "#57606a", "#afb8c1"
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200)
    ax.plot([0, 1], [0, 1], color=muted, lw=1.2, ls=(0, (4, 4)), label="perfect calibration")
    def line(rel, color, label, ls="-", alpha=0.35):
        rel = [r for r in rel if r["n"] >= 20]
        xs, ys = [r["conf"] for r in rel], [r["acc"] for r in rel]
        ax.plot(xs, ys, color=color, lw=2, ls=ls, label=label, zorder=2)
        ax.scatter(xs, ys, s=[6 + r["n"] / 6 for r in rel], color=color, alpha=alpha, edgecolors=color, zorder=3)
    s, f, fT = res["stream"]["choice"], res["fixed"]["choice"], res["fixed"]["choice_T"]
    line(fT["reliability"], light, f"same, rescaled (T = {res['fixed']['T_option']:.0f})  ECE {fT['ece']:.3f}, {100 * fT['acc']:.1f}% correct", ls="--", alpha=0.25)
    line(f["reliability"], dark, f"512-ticket pool, resampled  ECE {f['ece']:.3f}, {100 * f['acc']:.1f}% correct")
    line(s["reliability"], c1, f"fresh synthetic stream  ECE {s['ece']:.3f}, {100 * s['acc']:.1f}% correct")
    big = max((r for r in f["reliability"]), key=lambda r: r["n"])
    ax.annotate(f"{big['n']:,} answers said {big['conf']:.2f};\n{100 * big['acc']:.0f}% of them were right",
                xy=(big["conf"], big["acc"]), xytext=(0.62, 0.40), fontsize=10, color=ink, ha="left")
    ax.set_xlabel("top probability the model gave (binned)"); ax.set_ylabel("share actually correct")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0); ax.grid(color=grid, lw=0.6, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("Low calibration error can hide weak predictions", fontsize=13, color=ink, loc="left")
    fig.tight_layout(); fig.savefig(output_dir / "reliability.png"); plt.close(fig)

    return


def token_level_mask_figure():
    """Token-level attention mask for one synthetic request."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink = "#1f2328"
    # Attention mask for one small request.
    r = D.make_rng(77)
    req = D.sample_request(r, n_opts=(2, 2), name_mode="topic")
    req["questions"] = req["questions"][:3]
    b = D.collate([req]); allowed = b["mask_bias"][0, 0] == 0
    enc = D.encode_request(req)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)
    ax.imshow(allowed, cmap="Blues", vmin=0, vmax=1.4, interpolation="nearest")
    # block boundaries and labels
    kinds, qids, oids = np.array(enc["kind"]), np.array(enc["qid"]), np.array(enc["oid"])
    keys = list(zip(kinds, qids, oids))
    starts = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    names = []
    for a0 in starts[:-1]:
        k, q, o = keys[a0]
        qt = req["questions"][q]["type"] if q >= 0 else ""
        if k == D.STATE:
            names.append("state")
        elif k == D.QUESTION:
            names.append(f"{qt} question")
        else:
            names.append(f"score level {o}" if qt == "score" else f"{qt} option {o + 1}")
    for s_ in starts[1:-1]:
        ax.axhline(s_ - 0.5, color="white", lw=1.2); ax.axvline(s_ - 0.5, color="white", lw=1.2)
    mids = [(a0 + a1 - 1) / 2 for a0, a1 in zip(starts[:-1], starts[1:])]
    ax.set_yticks(mids); ax.set_yticklabels(names, fontsize=11)
    ax.set_xticks(mids); ax.set_xticklabels(names, fontsize=11, rotation=90)
    ax.set_xlabel("key (what a token may read)"); ax.set_ylabel("query (the token doing the reading)")
    ax.set_title("Dark = may attend. One sequence, independent answers.", fontsize=11, color=ink, loc="left")
    fig.tight_layout(); fig.savefig("figures/attention_mask.png"); plt.close(fig)


if __name__ == "__main__":
    main()
