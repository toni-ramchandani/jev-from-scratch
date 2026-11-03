"""Show a restriction of this toy scorer, not of Jev.

Adding sibling options cannot change score(A)-score(B) for shared candidates.
The generator's exact posterior requires this log-odds ratio to depend on n.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import data as D
from evaluate import load


def probe(model):
    topics = D.TRAIN_TOPICS[:4]
    outputs = []
    for n in (2, 4):
        q = {"type": "choice", "instr": D.CHOICE_Q,
             "options": [(t, D.TOPICS[t]) for t in topics[:n]],
             "option_ids": topics[:n], "target": [1.] + [0.] * (n - 1)}
        # These cues and fillers have positive probability under both option
        # sets. Unrelated urgency/flag words cannot reveal the option count.
        state = [D.TOPICS[topics[0]][0], D.URGENCY[0][1][0],
                 D.FLAGS["angry"][1][0]] + D.FILLERS[:2]
        req = {"state": state, "questions": [q]}
        scores, _ = model.forward(D.collate([req]))
        log_odds = float(scores.data[0] - scores.data[1])
        outputs.append({"options": n, "model_log_odds_A_over_B": log_odds,
                        "model_odds_A_over_B": float(np.exp(log_odds)),
                        "bayes_odds_A_over_B": D.Q_TOPIC * (n - 1) / (1 - D.Q_TOPIC)})
    return {"note": "Same state and shared candidates; only sibling options change",
            "cases": outputs,
            "model_abs_log_odds_change": abs(outputs[0]["model_log_odds_A_over_B"] - outputs[1]["model_log_odds_A_over_B"]),
            "conclusion": "This scorer cannot exactly represent both Bayes posteriors. No numerical performance ceiling is established."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("../artifacts/supplied"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = probe(load("stream", args.checkpoint_dir))
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
