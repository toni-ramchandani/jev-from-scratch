"""data.py: a synthetic support-desk world with known answers *and* known probabilities.

Each request is a short "ticket" (the state) plus questions in TypeSafe's three
documented shapes: Choice (pick one option), Score (ordered levels, expected
value) and Noul (probability of true). Cue words are sampled noisily from hidden
labels, so the Bayes-optimal probability of every answer has a closed form.
Real data never gives you that; here it lets us check calibration against truth.
"""
import random
import numpy as np

# ---- the world ------------------------------------------------------------
TOPICS = {  # option name -> cue words (used in tickets and in option descriptions)
    "billing":   ["invoice", "charged", "refund", "payment"],
    "shipping":  ["delivery", "tracking", "courier", "parcel"],
    "returns":   ["exchange", "defective", "swap", "faulty"],
    "login":     ["password", "locked", "signin", "reset"],
    "privacy":   ["gdpr", "delete-data", "consent", "tracking-cookies"],
    "pricing":   ["discount", "coupon", "quote", "plan-cost"],
    "api":       ["endpoint", "token", "rate-limit", "webhook"],
    "mobile":    ["android", "iphone", "app-crash", "push"],
    "hardware":  ["router", "cable", "firmware", "led"],
    "sales":     ["demo", "enterprise", "trial", "seats"],
    "legal":     ["contract", "liability", "clause", "terms"],
    "careers":   ["job", "resume", "interview", "hiring"],
    # held out: never offered during training (zero-shot test only)
    "warranty":  ["guarantee", "cover", "claim", "expiry"],
    "events":    ["webinar", "conference", "booth", "ticket"],
    "accessibility": ["screen-reader", "contrast", "captions", "keyboard-nav"],
    "partners":  ["reseller", "affiliate", "referral", "co-marketing"],
}
HELD_OUT = ["warranty", "events", "accessibility", "partners"]
TRAIN_TOPICS = [t for t in TOPICS if t not in HELD_OUT]

URGENCY = [  # Score levels 0, 1, 2 (ordered)
    ("low", ["whenever", "no-rush", "eventually"]),
    ("medium", ["soon", "this-week", "follow-up"]),
    ("high", ["asap", "urgent", "immediately"]),
]
FLAGS = {  # Noul questions: instruction words, true-cues, false-cues
    "angry":   (["is", "the", "customer", "angry"], ["furious", "annoyed", "outraged"], ["calm", "pleased", "relaxed"]),
    "repeat":  (["is", "this", "a", "repeat", "contact"], ["again", "second-time", "still-waiting"], ["first-time", "new-issue", "just-noticed"]),
    "human":   (["does", "the", "customer", "want", "a", "human"], ["call-me", "real-person", "agent"], ["email-ok", "chat-ok", "self-serve"]),
    "vip":     (["is", "the", "customer", "high", "value"], ["premium", "gold-tier", "key-account"], ["free-tier", "basic", "hobby"]),
}
FILLERS = ["hi", "hello", "please", "help", "my", "order", "today", "thanks", "need", "with", "account", "issue", "question", "regarding"]
ALIASES = [f"team-{c}" for c in "abcdefghijklmnop"]  # training-time option names
CHOICE_Q = ["which", "team", "should", "handle", "this"]
SCORE_Q = ["how", "urgent", "is", "this"]
OTHER_DESC = ["none-fits", "unlisted"]

Q_TOPIC, Q_URGENT, Q_FLAG = 0.7, 0.8, 0.75  # probability a cue word is "honest"

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[CHOICE]", "[SCORE]", "[NOUL]", "[OPT]", ":", "[TRUE]", "[FALSE]"]


def build_vocab():
    words = list(SPECIALS)
    for name, cues in TOPICS.items():
        words += [name] + cues
    for name, cues in URGENCY:
        words += [name] + cues
    for instr, yes, no in FLAGS.values():
        words += instr + yes + no
    words += FILLERS + ALIASES + CHOICE_Q + SCORE_Q + ["other"] + OTHER_DESC
    vocab = {}
    for w in words:
        vocab.setdefault(w, len(vocab))
    return vocab


VOCAB = build_vocab()
STATE, QUESTION, OPTION, PAD = 0, 1, 2, 3
MAX_STATE, MAX_Q, MAX_OPT = 40, 24, 12  # position offsets: questions/options share position ranges
N_POS = MAX_STATE + MAX_Q + MAX_OPT


# ---- sampling one request -------------------------------------------------
def sample_request(rng, n_opts=(2, 6), topics=None, oos=False, offer_other=False,
                   name_mode="mixed"):
    """Return a request dict with tokens per block, targets and Bayes posteriors.

    oos=True: the true topic is NOT among the offered options (forced-choice test).
    offer_other: add an "other" option; if oos, "other" is the correct answer.
    name_mode: "mixed" (training), "topic" (real names), "hidden" (all [UNK]).
    """
    topics = topics or TRAIN_TOPICS
    n = rng.randint(*n_opts)
    offered = rng.sample(topics, n)
    if oos:
        pool = [t for t in topics if t not in offered]
        true_topic = rng.choice(pool)
    else:
        true_topic = rng.choice(offered)

    state = []
    # topic cues: honest with prob Q_TOPIC, else a distractor from another offered option
    m = rng.randint(1, 3)
    for _ in range(m):
        if rng.random() < Q_TOPIC:
            src = true_topic
        else:
            src = rng.choice([t for t in offered if t != true_topic])
        state.append(rng.choice(TOPICS[src]))
    # urgency cues (Score)
    level = rng.randrange(3)
    for _ in range(rng.randint(1, 2)):
        src = level if rng.random() < Q_URGENT else rng.choice([l for l in range(3) if l != level])
        state.append(rng.choice(URGENCY[src][1]))
    # flag cues (Noul), 1-2 flags asked per request
    flag_names = rng.sample(list(FLAGS), rng.randint(1, 2))
    flag_vals = {}
    for f in flag_names:
        val = rng.random() < 0.5
        flag_vals[f] = val
        _, yes, no = FLAGS[f]
        for _ in range(rng.randint(1, 2)):
            honest = rng.random() < Q_FLAG
            side = yes if (val == honest) else no
            state.append(rng.choice(side))
    state += rng.sample(FILLERS, rng.randint(2, 4))
    rng.shuffle(state)

    def option_name(t):
        if name_mode == "topic":
            return t
        if name_mode == "hidden":
            return "[UNK]"
        r = rng.random()
        return t if r < 0.5 else (rng.choice(ALIASES) if r < 0.8 else "[UNK]")

    options = [(option_name(t), TOPICS[t]) for t in offered]
    target = [1.0 if t == true_topic else 0.0 for t in offered]
    if offer_other:
        options.append(("other", OTHER_DESC))
        target.append(1.0 if oos else 0.0)

    # Bayes-optimal posterior over offered topics (in-schema, no "other"):
    # P(t | cues) ∝ r^{k_t}, r = q(n-1)/(1-q), k_t = cue words from topic t.
    if not oos:
        k = np.array([sum(w in TOPICS[t] for w in state) for t in offered], float)
        r = Q_TOPIC * (n - 1) / (1 - Q_TOPIC)
        post = r ** k
        bayes_choice = post / post.sum()
    else:
        bayes_choice = None
    # urgency posterior ∝ (0.8/0.1)^{k_level}
    ku = np.array([sum(w in URGENCY[l][1] for w in state) for l in range(3)], float)
    pu = (Q_URGENT / ((1 - Q_URGENT) / 2)) ** ku
    bayes_score = pu / pu.sum()
    bayes_flags = {}
    for f in flag_names:
        _, yes, no = FLAGS[f]
        d = sum(w in yes for w in state) - sum(w in no for w in state)
        odds = (Q_FLAG / (1 - Q_FLAG)) ** d
        bayes_flags[f] = odds / (1 + odds)

    questions = [
        {"type": "choice", "instr": CHOICE_Q, "options": options, "target": target,
         "bayes": bayes_choice, "offered": offered, "true_topic": true_topic},
        {"type": "score", "instr": SCORE_Q,
         "options": [(name, cues) for name, cues in URGENCY],
         "target": [1.0 if l == level else 0.0 for l in range(3)], "bayes": bayes_score},
    ]
    for f in flag_names:
        instr, yes, no = FLAGS[f]
        questions.append({"type": "noul", "instr": instr + ["[TRUE]"] + yes + ["[FALSE]"] + no,
                          "target": float(flag_vals[f]), "bayes": bayes_flags[f], "flag": f})
    return {"state": state, "questions": questions}


# ---- packing many questions into ONE sequence -------------------------------
def encode_request(req):
    """Flatten a request into one token sequence plus the metadata the
    block attention mask needs. Questions and options get *fixed* position
    offsets, so the order in which they are packed does not change positions."""
    ids, pos, seg, kind, qid, oid = [], [], [], [], [], []
    def add(tokens, k, q, o, start):
        for i, w in enumerate(tokens):
            ids.append(VOCAB.get(w, VOCAB["[UNK]"])); pos.append(start + i)
            seg.append(k); kind.append(k); qid.append(q); oid.append(o)

    add(["[CLS]"] + req["state"], STATE, -1, -1, 0)
    readouts = []  # (question index, option index or -1, token index)
    for qi, q in enumerate(req["questions"]):
        marker = {"choice": "[CHOICE]", "score": "[SCORE]", "noul": "[NOUL]"}[q["type"]]
        readouts.append((qi, -1, len(ids)))
        add([marker] + q["instr"], QUESTION, qi, -1, MAX_STATE)
        for oi, (name, desc) in enumerate(q.get("options", [])):
            readouts.append((qi, oi, len(ids)))
            add(["[OPT]", name, ":"] + list(desc), OPTION, qi, oi, MAX_STATE + MAX_Q)
    return dict(ids=ids, pos=pos, seg=seg, kind=kind, qid=qid, oid=oid, readouts=readouts)


def collate(reqs):
    """Pad a list of requests into arrays and build the (B, T, T) attention mask."""
    encs = [encode_request(r) for r in reqs]
    T = max(len(e["ids"]) for e in encs)
    B = len(encs)
    ids = np.zeros((B, T), int); pos = np.zeros((B, T), int); seg = np.zeros((B, T), int)
    kind = np.full((B, T), PAD); qid = np.full((B, T), -9); oid = np.full((B, T), -9)
    for b, e in enumerate(encs):
        L = len(e["ids"])
        ids[b, :L], pos[b, :L], seg[b, :L] = e["ids"], e["pos"], e["seg"]
        kind[b, :L], qid[b, :L], oid[b, :L] = e["kind"], e["qid"], e["oid"]
    seg = np.minimum(seg, 2)
    ki, kj = kind[:, :, None], kind[:, None, :]
    same_q = qid[:, :, None] == qid[:, None, :]
    same_o = oid[:, :, None] == oid[:, None, :]
    allowed = np.broadcast_to(kj == STATE, (B, T, T)).copy()           # everyone reads the state
    allowed |= (ki != STATE) & (ki != PAD) & (kj == QUESTION) & same_q  # own question
    allowed |= (ki == OPTION) & (kj == OPTION) & same_q & same_o        # own option only
    mask_bias = np.where(allowed, 0.0, -1e9)[:, None, :, :]           # (B, 1, T, T)

    # readout indices and loss targets
    opt_rows, opt_group, opt_target, noul_rows, noul_target = [], [], [], [], []
    meta = []  # one entry per question, for evaluation
    g = 0
    for b, (e, r) in enumerate(zip(encs, reqs)):
        q_rows = {}
        for qi, oi, t in e["readouts"]:
            q_rows.setdefault(qi, []).append((oi, b * T + t))
        for qi, q in enumerate(r["questions"]):
            rows = q_rows[qi]
            if q["type"] == "noul":
                meta.append(("noul", len(noul_rows), q))
                noul_rows.append(rows[0][1]); noul_target.append(q["target"])
            else:
                opts = [row for oi, row in rows if oi >= 0]
                meta.append((q["type"], (len(opt_rows), len(opt_rows) + len(opts), g), q))
                opt_rows += opts; opt_group += [g] * len(opts); opt_target += q["target"]
                g += 1
    return dict(ids=ids, pos=pos, seg=seg, mask_bias=mask_bias,
                opt_rows=np.array(opt_rows, dtype=int), opt_group=np.array(opt_group, dtype=int),
                opt_target=np.array(opt_target, dtype=float), n_groups=g,
                noul_rows=np.array(noul_rows, dtype=int), noul_target=np.array(noul_target, dtype=float),
                meta=meta)


def make_rng(seed):
    return random.Random(seed)
