"""Small local adapter for text-only Choice, Score and Noul payloads.

This is a subset of the public request idea, not a Jev API server or compatible
SDK. It uses the toy's fixed word vocabulary and short position ranges. It
does not contact TypeSafe, implement billing, or understand arbitrary language.
"""
import argparse
import json
from pathlib import Path
import data as D
from evaluate import load

MODEL_ID = "tiny-s1-experimental"


def request_to_internal(payload):
    if not isinstance(payload.get("state"), str):
        raise ValueError("The toy adapter accepts a text state only")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty named map")
    internal = []
    for name, q in questions.items():
        if not isinstance(name, str) or not isinstance(q, dict):
            raise ValueError("Each question needs a string id and an object")
        typ, instr = q.get("type"), q.get("instructions")
        if typ not in ("choice", "score", "noul") or not isinstance(instr, str):
            raise ValueError("Use choice, score or noul with text instructions")
        item = {"type": typ, "instr": instr.split(), "bayes": None}
        criteria = q.get("criteria")
        if typ == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice needs 1 to 255 named options")
            if any(not isinstance(k, str) or (v is not None and not isinstance(v, str))
                   for k, v in criteria.items()):
                raise ValueError("Choice names and descriptions must be text")
            item["options"] = [(k, (v or "").split()) for k, v in criteria.items()]
            item["option_ids"] = list(criteria)
            item["target"] = [0.0] * len(criteria)  # prediction has no labels
        elif typ == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score needs 2 to 10 ordered descriptions")
            if any(not isinstance(v, str) for v in criteria):
                raise ValueError("Score level descriptions must be text")
            # A leading 'low:' is a toy tokenization convention, not a vendor rule.
            opts = []
            for i, text in enumerate(criteria):
                label, sep, description = text.partition(":")
                opts.append((label.strip() if sep else str(i),
                             (description if sep else text).split()))
            item["options"] = opts
            item["target"] = [0.0] * len(opts)
        else:
            item["target"] = 0.0
            if criteria is not None:
                if (not isinstance(criteria, dict) or set(criteria) != {"true", "false"}
                        or any(not isinstance(v, str) for v in criteria.values())):
                    raise ValueError("Noul criteria need true and false text descriptions")
                item["instr"] += ["[TRUE]"] + criteria["true"].split()
                item["instr"] += ["[FALSE]"] + criteria["false"].split()
        internal.append(item)
    return {"state": payload["state"].split(), "questions": internal}


def predict(model, payload):
    request = request_to_internal(payload)
    answers = model.answer(D.collate([request]))
    return {"model": MODEL_ID, "answers": dict(zip(payload["questions"], answers))}


def example_payload():
    return {
        "model": MODEL_ID,
        "state": "help hi furious thanks seats eventually seats account hobby",
        "questions": {
            "team": {"type": "choice", "instructions": "which team should handle this",
                     "criteria": {"returns": "exchange defective swap faulty",
                                  "billing": "invoice charged refund payment",
                                  "sales": "demo enterprise trial seats",
                                  "api": "endpoint token rate-limit webhook",
                                  "hardware": "router cable firmware led"}},
            "urgency": {"type": "score", "instructions": "how urgent is this",
                        "criteria": ["low: whenever no-rush eventually",
                                     "medium: soon this-week follow-up",
                                     "high: asap urgent immediately"]},
            "angry": {"type": "noul", "instructions": "is the customer angry",
                      "criteria": {"true": "furious annoyed outraged", "false": "calm pleased relaxed"}},
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("../artifacts/supplied"))
    ap.add_argument("--run", choices=["stream", "fixed", "other"], default="stream")
    ap.add_argument("--request", type=Path)
    args = ap.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8")) if args.request else example_payload()
    print(json.dumps(predict(load(args.run, args.checkpoint_dir), payload), indent=2, allow_nan=False))
