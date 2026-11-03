"""model.py: a small transformer encoder that answers typed questions in one pass.

Contract (mirrors TypeSafe's documented interface, not its undisclosed internals):
  * the caller supplies the answer space (option names + descriptions);
  * every option gets one scalar score; a softmax *within each question*
    turns scores into normalized probabilities; Choice selects a supplied
    option and Score reports the expected level;
  * all questions about one state are packed into one sequence and answered
    in a single forward pass; a block attention mask keeps them independent.
"""
import numpy as np
from autograd import Tensor, gelu, softmax, layer_norm, take_rows, DTYPE
import autograd as ag
import data as D


def param(*shape, scale=0.02, rng=None):
    return Tensor(rng.normal(0, scale, size=shape), requires_grad=True)


class TinyS1:
    def __init__(self, vocab_size, d=64, n_heads=4, n_layers=2, d_ff=256, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.h = d, n_heads
        self.tok = param(vocab_size, d, rng=rng)
        self.pos = param(D.N_POS, d, rng=rng)
        self.seg = param(3, d, rng=rng)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(dict(
                ln1_g=Tensor(np.ones(d), requires_grad=True), ln1_b=Tensor(np.zeros(d), requires_grad=True),
                w_qkv=param(d, 3 * d, rng=rng), b_qkv=Tensor(np.zeros(3 * d), requires_grad=True),
                w_o=param(d, d, rng=rng, scale=0.02 / np.sqrt(2 * n_layers)), b_o=Tensor(np.zeros(d), requires_grad=True),
                ln2_g=Tensor(np.ones(d), requires_grad=True), ln2_b=Tensor(np.zeros(d), requires_grad=True),
                w_1=param(d, d_ff, rng=rng), b_1=Tensor(np.zeros(d_ff), requires_grad=True),
                w_2=param(d_ff, d, rng=rng, scale=0.02 / np.sqrt(2 * n_layers)), b_2=Tensor(np.zeros(d), requires_grad=True),
            ))
        self.lnf_g = Tensor(np.ones(d), requires_grad=True)
        self.lnf_b = Tensor(np.zeros(d), requires_grad=True)
        self.w_opt = param(d, 1, rng=rng)      # option score head (Choice and Score)
        self.w_noul = param(d, 1, rng=rng)     # Noul head: logit of "true"
        self.b_noul = Tensor(np.zeros(1), requires_grad=True)

    def parameters(self):
        ps = [self.tok, self.pos, self.seg, self.lnf_g, self.lnf_b, self.w_opt, self.w_noul, self.b_noul]
        for L in self.layers:
            ps += list(L.values())
        return ps

    def matrices(self):  # weight decay applies to these only (not embeddings, norms, biases)
        return [L[k] for L in self.layers for k in ("w_qkv", "w_o", "w_1", "w_2")]

    def attention(self, x, L, mask_bias):
        B, T, d = x.shape
        h, dh = self.h, d // self.h
        qkv = x @ L["w_qkv"] + L["b_qkv"]                       # (B, T, 3d)
        qkv = qkv.reshape(B, T, 3, h, dh).transpose(2, 0, 3, 1, 4)  # (3, B, h, T, dh)
        q, k, v = ag.index0(qkv, 0), ag.index0(qkv, 1), ag.index0(qkv, 2)
        q = q * float(1.0 / np.sqrt(dh))           # scale q, the smaller tensor
        att = q @ k.transpose(0, 1, 3, 2) + mask_bias   # blocked pairs get -1e9
        att = softmax(att, axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        return y @ L["w_o"] + L["b_o"]

    def encode(self, batch):
        x = take_rows(self.tok, batch["ids"]) + take_rows(self.pos, batch["pos"]) + take_rows(self.seg, batch["seg"])
        mb = Tensor(batch["mask_bias"])
        for L in self.layers:
            x = x + self.attention(layer_norm(x, L["ln1_g"], L["ln1_b"]), L, mb)
            hdn = gelu(layer_norm(x, L["ln2_g"], L["ln2_b"]) @ L["w_1"] + L["b_1"])
            x = x + (hdn @ L["w_2"] + L["b_2"])
        x = layer_norm(x, self.lnf_g, self.lnf_b)
        B, T, d = x.shape
        return x.reshape(B * T, d)

    def forward(self, batch):
        hflat = self.encode(batch)
        opt_scores = (take_rows(hflat, batch["opt_rows"]) @ self.w_opt).reshape(-1)
        noul_logits = (take_rows(hflat, batch["noul_rows"]) @ self.w_noul + self.b_noul).reshape(-1)
        return opt_scores, noul_logits

    def loss(self, batch):
        s, z = self.forward(batch)
        l_opt = ag.grouped_softmax_xent(s, batch["opt_group"], batch["opt_target"], batch["n_groups"])
        l_noul = ag.bce_with_logits(z, batch["noul_target"])
        return l_opt + l_noul, (s, z)

    # ---- inference in TypeSafe's response shape ----------------------------
    def answer(self, batch, t_opt=1.0, t_noul=1.0):
        if not np.isfinite(t_opt) or not np.isfinite(t_noul) or min(t_opt, t_noul) <= 0:
            raise ValueError("Temperatures must be finite and positive")
        s, z = self.forward(batch)
        p_opt = ag.group_softmax_np(s.data.astype(float) / t_opt, batch["opt_group"], batch["n_groups"])
        p_noul = ag.sigmoid_np(z.data.astype(float) / t_noul)
        out = []
        for typ, ref, q in batch["meta"]:
            if typ == "noul":
                out.append({"type": "noul", "noul": float(p_noul[ref])})
                continue
            a, b, _ = ref
            p = p_opt[a:b]
            names = q.get("option_ids", [o[0] for o in q["options"]])
            if typ == "choice" and len(set(names)) != len(names):
                raise ValueError("Duplicate visible names need unique option_ids")
            ans = {"type": typ, "probabilities": dict(zip(map(str, range(len(p))) if typ == "score" else names, p.tolist())),
                   "confidence": float(confidence(p))}
            if typ == "choice":
                ans["choice"] = names[int(p.argmax())]
            else:  # Score: expected value over ordered levels, as documented
                ans["score"] = float((p * np.arange(len(p))).sum())
                ans["legend"] = {str(i): name + ": " + " ".join(desc)
                                 for i, (name, desc) in enumerate(q["options"])}
            out.append(ans)
        return out


def confidence(p):
    """1 - normalized entropy. TypeSafe documents confidence as a function of how
    the probabilities are spread but does not publish its formula; this is ours."""
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    if len(p) < 2:
        return 1.0
    return 1.0 + (p * np.log(p)).sum() / np.log(len(p))
