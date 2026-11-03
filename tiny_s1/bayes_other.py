"""bayes_other.py: what would a perfect model do when an `other` option is offered?

Uses the 'other' run's training prior (when `other` is offered, the right team is
missing half the time) and the generator's exact likelihoods. Prints how often the
Bayes-optimal decision picks `other` on the same test sets evaluate.py uses.
Result on 2026-11-02: no-fit tickets 0.867, tickets where a listed team fits 0.000.
"""
import numpy as np
import data as D

Q = D.Q_TOPIC


def source_topic(word):
    return next(t for t in D.TRAIN_TOPICS if word in D.TOPICS[t])


def posterior(req):
    q = req["questions"][0]
    offered = q["offered"]
    n = len(offered)
    cues = [w for w in req["state"] if any(w in D.TOPICS[t] for t in D.TRAIN_TOPICS)]
    unoffered = [t for t in D.TRAIN_TOPICS if t not in offered]
    like_in = []
    for t in offered:  # right team is listed
        p = 1.0
        for w in cues:
            s = source_topic(w)
            p *= Q / 4 if s == t else ((1 - Q) / ((n - 1) * 4) if s in offered else 0.0)
        like_in.append(p / n)
    like_out = 0.0
    for t in unoffered:  # right team is missing
        p = 1.0
        for w in cues:
            s = source_topic(w)
            p *= Q / 4 if s == t else ((1 - Q) / (n * 4) if s in offered else 0.0)
        like_out += p / len(unoffered)
    post = np.array(like_in + [like_out]) * 0.5
    return post / post.sum()


if __name__ == "__main__":
    for name, seed, oos in [("no-fit, other offered", 5006, True), ("fits, other offered", 5007, False)]:
        r = D.make_rng(seed)
        reqs = [D.sample_request(r, oos=oos, offer_other=True) for _ in range(2000)]
        picks = np.mean([posterior(x).argmax() == len(x["questions"][0]["offered"]) for x in reqs])
        print(f"Bayes, {name}: picks other {picks:.3f}")
