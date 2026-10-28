"""Step 9: the best public file's levels, with the ensemble's ranking inside each segment.

The best known public score (0.16529, ``data/reference``) comes from one transfer LightGBM with
new-business and gender factors that were settled with leaderboard probes. Step 8's ensemble ranks
risks better on the 2024 back-test, but its own levels lost on the leaderboard (v4, 0.16198):
buildings 6% lower than the best file, motor renewals 7% higher.

So this step keeps the best file's total in every segment (line x new business x gender for motor
and contents, the lines where it applies gender factors; line x new business for buildings) and
changes only the ranking inside each segment: a geometric blend of the best file and step 8's
chosen experts, with weight ``--weight`` on the best file.

Check: the same blend on the 2024 back-test, each segment rescaled to its actual claims (level
known), rows weighted to the 2025 mix (77% new business, against 13-15% in 2024).

Output: the next version in ``submissions/`` (``v<N>_best_levels_ranked_w<weight>.csv``) and
``reports/09_rebase/``.

Usage: python 09_rebase_to_best.py [--weight 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import geo_experts as G  # noqa: E402
import pricing_lgbm as P  # noqa: E402
import submissions as S  # noqa: E402

REF = C.DATA / "reference"
EXPERTS_DIR = C.DATA / "experts"
OUT = C.REPORTS / "09_rebase"
GENDER_LINES = ("motor", "contents")    # the best file's gender factors split these lines' segments
NB_SHARE_2025 = 0.77                    # new business in the 2025 book


def segments(df: pd.DataFrame, line: str) -> np.ndarray:
    """Segment of each row: new business or renewal, and gender where the best file prices it."""
    seg = np.where(df["_new_business"].to_numpy(dtype=bool), "new", "renewal").astype(object)
    if line in GENDER_LINES:
        seg = seg + "_" + df["_gender"].astype(str).to_numpy()
    return seg


def blend(best: np.ndarray, ours: np.ndarray, w: float) -> np.ndarray:
    """Geometric blend: w on the best file's ranking, 1 - w on the ensemble's."""
    return np.exp(w * np.log(best) + (1 - w) * np.log(ours))


def rebase(pred: np.ndarray, target: np.ndarray, seg: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Rescale ``pred`` so each segment's (weighted) total equals ``target``'s."""
    w = np.ones(len(pred)) if weight is None else weight
    out = pred.astype(float).copy()
    for s in np.unique(seg):
        sel = seg == s
        out[sel] *= np.sum(w[sel] * target[sel]) / np.sum(w[sel] * pred[sel])
    return out


def backtest(line: str, members: list[str], weight: float) -> dict[str, tuple[float, float]]:
    """Weighted deviance and null deviance of each ranking on 2024, every segment at its actual level."""
    ref = pd.read_parquet(REF / f"best_ho2024_{line}.parquet")
    ours = pd.read_parquet(EXPERTS_DIR / f"{line}_ho2024.parquet")
    m = ref.merge(ours[["row_id", "_new_business", "_gender", *members]], on="row_id", validate="one_to_one")
    y = m["y"].to_numpy(dtype=float)
    share = m["comp_line"].fillna(1.0).clip(1e-4, 1.0).to_numpy()          # reported basis, as the claims
    nb = m["_new_business"].to_numpy(dtype=bool)
    w = np.where(nb, NB_SHARE_2025 / nb.mean(), (1 - NB_SHARE_2025) / (~nb).mean())   # 2025 mix
    seg = segments(m, line)
    factors = json.loads((REF / "best_gender_factors.json").read_text(encoding="utf-8"))
    gf = m["_gender"].astype(str).map(factors[line]).to_numpy(dtype=float) if line in GENDER_LINES else 1.0
    best = m["tr_gbm"].to_numpy(dtype=float) * gf
    mine = G.geo_mean(m, members)
    null = 0.0
    for s in np.unique(seg):                                               # one price per segment
        sel = seg == s
        null += float(np.sum(w[sel] * P.tweedie_deviance(y[sel], np.full(sel.sum(), np.average(y[sel], weights=w[sel])))))
    out = {}
    for name, mu in (("best file", best), ("ensemble", mine), ("blend", blend(best, mine, weight))):
        mu = rebase(mu * share, y, seg, w)
        out[name] = (float(np.sum(w * P.tweedie_deviance(y, mu))), null)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight", type=float, default=0.5, help="weight of the best file's ranking (0 to 1)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    members = json.loads((C.REPORTS / "08_geo" / "summary.json").read_text(encoding="utf-8"))["members"]
    best = pd.read_csv(REF / "best_public_gbm100_submission.csv", dtype={"row_id": str}).set_index("row_id")["prediction"]
    finals, report, pooled = [], [], {}
    for line in C.LINES:
        fin = pd.read_parquet(EXPERTS_DIR / f"{line}_final.parquet")
        ref = best.loc[fin["row_id"].astype(str)].to_numpy(dtype=float)
        seg = segments(fin, line)
        pred = rebase(blend(ref, G.geo_mean(fin, members[line]), a.weight), ref, seg)   # best file's segment totals
        finals.append(pd.DataFrame({"row_id": fin["row_id"].astype(str).to_numpy(), "prediction": pred}))
        bt = backtest(line, members[line], a.weight)
        for k, v in bt.items():
            pooled.setdefault(k, []).append(v)
        d2 = {k: 1 - dev / null for k, (dev, null) in bt.items()}
        corr = float(np.corrcoef(np.log(pred), np.log(ref))[0, 1])
        report.append({"line": line, "members": "+".join(members[line]), **{f"d2_{k}": v for k, v in d2.items()},
                       "log_corr_with_best": corr, "total_vs_best": float(pred.sum() / ref.sum())})
        print(f"{line:<9} 2024 ranking D2 (level known per segment): best file {d2['best file']:.4f}, "
              f"ensemble {d2['ensemble']:.4f}, blend {d2['blend']:.4f}; log corr with best {corr:.3f}")
    d2 = {k: 1 - sum(v[0] for v in parts) / sum(v[1] for v in parts) for k, parts in pooled.items()}
    print(f"all lines  best file {d2['best file']:.4f}, ensemble {d2['ensemble']:.4f}, blend {d2['blend']:.4f}")

    expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = pd.concat(finals, ignore_index=True).set_index("row_id").loc[expected.to_numpy()].reset_index()
    for name, ok in P.check_submission(sub, expected).items():   # raises if any check fails
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    tag = f"w{int(round(a.weight * 100)):02d}"
    path = S.save(sub, C.ROOT, f"best_levels_ranked_{tag}", "09_rebase_to_best.py",
                  f"best public file's total in every segment (line x new business x gender); ranking inside: "
                  f"{a.weight:.2f} best file + {1 - a.weight:.2f} step-8 ensemble",
                  f"2024 ranking D2 within segments {d2['blend']:.4f} vs {d2['best file']:.4f} for the best file "
                  f"(rows weighted to 2025)")
    pd.DataFrame(report).to_csv(OUT / f"rebase_{tag}.csv", index=False)
    print(f"{len(sub):,} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
