"""Validate actual checkpoint responses and compare re-evaluated archive results."""
import json
from pathlib import Path
import numpy as np
import data as D
from evaluate import load


def main():
    root = Path(__file__).resolve().parents[1]
    report = {"checkpoints": {}, "reproduction_delta": {}}
    for source in ("supplied", "fresh"):
        for run in ("stream", "fixed", "other"):
            model = load(run, root / "artifacts" / source)
            r = D.make_rng(991)
            requests = [D.sample_request(r, name_mode="hidden") for _ in range(32)]
            batch = D.collate(requests)
            answers = model.answer(batch)
            worst = 0.
            for answer, (_, _, q) in zip(answers, batch["meta"]):
                if answer["type"] == "noul":
                    assert np.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1
                    continue
                p = answer["probabilities"]
                assert len(p) == len(q["options"])
                assert all(np.isfinite(v) and 0 <= v <= 1 for v in p.values())
                worst = max(worst, abs(sum(p.values()) - 1))
                if answer["type"] == "choice":
                    assert answer["choice"] in q["option_ids"]
                else:
                    assert abs(answer["score"] - sum(int(k) * v for k, v in p.items())) < 1e-12
            assert worst < 1e-12
            json.dumps(answers, allow_nan=False)
            report["checkpoints"][f"{source}/{run}"] = {
                "requests": 32, "answers": len(answers), "max_probability_sum_error": worst,
                "status": "passed"}
    old = json.loads((root / "artifacts/supplied/results.json").read_text())
    new = json.loads((root / "results/reproduced/results.json").read_text())
    for run in ("stream", "fixed", "other"):
        report["reproduction_delta"][run] = {
            key: new[run]["choice"][key] - old[run]["choice"][key]
            for key in ("acc", "ece", "nll", "brier")}
    report["note"] = "Stable log loss deliberately replaces epsilon-clipped evaluation. Archive training provenance is not independently established."
    (root / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
