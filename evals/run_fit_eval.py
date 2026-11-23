"""Evaluate the fit-scoring step against human-labeled cases.

    uv run python evals/run_fit_eval.py            # calls Claude: costs real tokens (~$0.05)
    uv run python evals/run_fit_eval.py --repeat 3 # measure run-to-run consistency

Reports exact-verdict accuracy, "ineligible" recall (the costly mistake is telling a founder
to apply to something they cannot win), and token cost. Results are saved under
evals/results/ so prompt changes can be compared run over run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.llm import ClaudeLLM  # noqa: E402
from scout.models import ExtractedOpportunity, FitAssessment, StartupProfile  # noqa: E402
from scout.opportunities import clamp_score  # noqa: E402
from scout.prompts import fit_system  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(HERE / "fit_cases.yaml"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", default="claude-opus-5-5")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))
    profile = StartupProfile.model_validate(spec["profile"])
    system = fit_system(profile)
    llm = ClaudeLLM(args.model)
    today = date(2026, 11, 23)  # fixed so labels stay valid as time passes

    rows, confusion = [], Counter()
    for case in spec["cases"]:
        opp = ExtractedOpportunity.model_validate(
            {"url": None, "equity_taken": None, "application_questions": [], **case["opportunity"]}
        )
        for run in range(args.repeat):
            fit = clamp_score(llm.structured(
                system=system,
                prompt=f"Today is {today.isoformat()}.\n\n<opportunity>\n"
                       f"{opp.model_dump_json(indent=2)}\n</opportunity>",
                schema=FitAssessment,
            ))
            ok = fit.verdict == case["expected"]
            confusion[(case["expected"], fit.verdict)] += 1
            rows.append({"id": case["id"], "run": run, "expected": case["expected"],
                         "got": fit.verdict, "score": fit.score, "ok": ok,
                         "blockers": fit.eligibility_blockers})
            print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<26} expected={case['expected']:<10} "
                  f"got={fit.verdict:<10} score={fit.score}")

    total = len(rows)
    accuracy = sum(r["ok"] for r in rows) / total
    inel = [r for r in rows if r["expected"] == "ineligible"]
    inel_recall = sum(r["got"] == "ineligible" for r in inel) / len(inel) if inel else None
    summary = {"model": args.model, "cases": total, "accuracy": round(accuracy, 3),
               "ineligible_recall": inel_recall, "usage": llm.usage.summary(),
               "confusion": {f"{e}->{g}": n for (e, g), n in sorted(confusion.items())}}
    print(json.dumps(summary, indent=2))

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"fit-{stamp}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
