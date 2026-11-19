"""Command-line prototype: score a commit feature record (Sec 11).

Examples
    python predict.py --example                      # score 10 held-out commits
    python predict.py --la 250 --ld 40 --nf 6 --nd 3 --ns 2 --ent 1.8 \\
                      --ndev 5 --age 30 --nuc 4 --aexp 120 --arexp 60 --asexp 30 --fix 0
    python predict.py --csv my_commits.csv           # one row per commit

The score is a research demonstration, not a claim that a defect will occur.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
MODEL = ROOT / "models" / "jit_model.joblib"


def load_bundle(path=MODEL):
    if not path.exists():
        sys.exit("model not found - run `python run_all.py` first")
    return joblib.load(path)


def score(bundle, records: pd.DataFrame) -> pd.DataFrame:
    feats = bundle["features"]
    missing = [f for f in feats if f not in records]
    if missing:
        raise ValueError(f"missing features: {missing}")
    if (records[feats] < 0).any().any():
        raise ValueError("feature values must be non-negative")
    prob = bundle["pipeline"].predict_proba(records[feats])[:, 1]
    hi, med = bundle["thresholds"]["high"], bundle["thresholds"]["medium"]
    level = pd.cut(prob, [-1, med, hi, 2], labels=["Low", "Medium", "High"], right=False)
    out = records.copy()
    out["risk_score"] = prob.round(4)
    out["risk_level"] = level.astype(str)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", action="store_true", help="score held-out examples")
    ap.add_argument("--csv", type=Path, help="CSV with one commit per row")
    for f in ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp",
              "asexp", "fix"]:
        ap.add_argument(f"--{f}", type=float)
    args = ap.parse_args()
    b = load_bundle()

    if args.example:
        recs = pd.read_csv(ROOT / "models" / "heldout_examples.csv").head(10)
    elif args.csv:
        recs = pd.read_csv(args.csv)
    else:
        vals = {f: getattr(args, f) for f in b["features"]}
        if any(v is None for v in vals.values()):
            ap.error("give all 13 features, --csv, or --example")
        recs = pd.DataFrame([vals])

    res = score(b, recs)
    cols = ["risk_score", "risk_level"] + (["buggy"] if "buggy" in res else [])
    print(res[["la", "ld", "nf"] + cols].to_string(index=False))
    print("\nthresholds:", json.dumps({k: round(v, 4) for k, v in b["thresholds"].items()}))
    print("model:", json.dumps(b["meta"]))


if __name__ == "__main__":
    main()
