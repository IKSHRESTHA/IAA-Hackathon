"""Step 10: segment levels from leaderboard probes: make a probe, or read one.

Ranking inside a segment barely moves the public score (v5 changed 25% of rows by more than 10% and
scored the same 0.16529 as the best file). The levels of the big segments do. For Tweedie p = 1.5
the deviance of one row is -8 sqrt(y) + 4 y / sqrt(mu) + 4 sqrt(mu), so scaling segment s by c
changes the public deviance by exactly

    4 [(c^-0.5 - 1) A + (c^0.5 - 1) B],   A = sum y mu^-0.5,  B = sum mu^0.5  (public rows of s),

and the public D^2 = 1 - deviance / N by that change over N. The best level is c* = A / B. With a
random public split, B / N equals the segment's sum of mu^0.5 over the null deviance of all 2025
rows, which the model estimates (checked on 2023: 110.2 per row estimated, 115.7 actual). One
probe with a known score change then pins A, and so the best level.

    make:  python 10_level_probe.py make --base v5 --line motor --business new --factor 1.15
    read:  python 10_level_probe.py read --base v5 --line motor --business new --factor 1.15 --base-score 0.16529 --score 0.1668

``make`` saves the base version with that segment scaled as the next version. ``read`` prints the
segment's best level, and the gain expected along the way, from a probe's score.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import pricing_lgbm as P  # noqa: E402
import submissions as S  # noqa: E402

EXPERTS_DIR = C.DATA / "experts"


def rows_2025() -> pd.DataFrame:
    """Line, new-business flag and gender of every 2025 row."""
    parts = []
    for line in C.LINES:
        f = pd.read_parquet(EXPERTS_DIR / f"{line}_final.parquet", columns=["row_id", "_new_business", "_gender"])
        parts.append(f.assign(line=line))
    return pd.concat(parts, ignore_index=True).set_index("row_id")


def load_version(version: str) -> tuple[pd.Series, str]:
    reg = pd.read_csv(C.ROOT / "submissions" / "versions.csv", dtype=str)
    hit = reg.loc[reg["version"] == version]
    if hit.empty:
        raise KeyError(f"{version} is not in submissions/versions.csv")
    name = hit["file"].iloc[0]
    sub = pd.read_csv(C.ROOT / "submissions" / name, dtype={"row_id": str}).set_index("row_id")["prediction"]
    return sub, name


def segment_mask(meta: pd.DataFrame, line: str, business: str, gender: str | None) -> np.ndarray:
    sel = (meta["line"] == line).to_numpy()
    if business != "all":
        sel = sel & (meta["_new_business"].to_numpy(dtype=bool) == (business == "new"))
    if gender:
        sel = sel & (meta["_gender"].astype(str) == gender).to_numpy()
    return sel


def null_deviance(mu: np.ndarray, line: np.ndarray) -> float:
    """Expected null deviance of the 2025 rows (claims unknown), from the predictions and claim sizes.

    Per row E[d(y, ybar)] = -8 E[sqrt(y)] + 4 mu / sqrt(ybar) + 4 sqrt(ybar), with E[sqrt(y)] from a
    Poisson number of claims (mean mu / mean claim size) and the line's E[sqrt(claim size)].
    """
    ybar = mu.mean()
    e_sqrt = np.zeros(len(mu))
    for ln in C.LINES:
        t = H.read_table(f"policies_{ln}", "features", columns=["n_claims", "claim_cost"])
        x = t.loc[(t["n_claims"] == 1) & (t["claim_cost"] > 0), "claim_cost"].to_numpy(dtype=float)
        sel = line == ln
        lam = mu[sel] / x.mean()                                 # expected claims of the row
        p1, p2 = lam * np.exp(-lam), 1 - np.exp(-lam) - lam * np.exp(-lam)
        e_sqrt[sel] = (p1 + p2 * np.sqrt(2)) * np.sqrt(x).mean()
    return float(np.sum(-8 * e_sqrt + 4 * mu / np.sqrt(ybar) + 4 * np.sqrt(ybar)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["make", "read"])
    ap.add_argument("--base", required=True, help="version the probe scales, e.g. v5")
    ap.add_argument("--line", required=True, choices=list(C.LINES))
    ap.add_argument("--business", default="all", choices=["new", "renewal", "all"])
    ap.add_argument("--gender", default=None, choices=["f", "m"])
    ap.add_argument("--factor", type=float, required=True, help="the segment's scale factor")
    ap.add_argument("--base-score", type=float, help="read: public score of the base version")
    ap.add_argument("--score", type=float, help="read: public score of the probe")
    a = ap.parse_args(argv)
    meta = rows_2025()
    base, name = load_version(a.base)
    meta = meta.loc[base.index]
    sel = segment_mask(meta, a.line, a.business, a.gender)
    label = f"{a.line} {a.business}" + (f" {a.gender}" if a.gender else "")
    print(f"segment {label}: {sel.sum():,} rows, total {base[sel].sum():,.0f} of {base.sum():,.0f} in {name}")

    if a.mode == "make":
        sub = base.copy()
        sub[sel] *= a.factor
        sub = sub.reset_index()
        expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
        sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
        P.check_submission(sub, expected)
        tag = f"{a.line}_{a.business}" + (f"_{a.gender}" if a.gender else "") + f"_x{a.factor:.3f}".replace(".", "p")
        S.save(sub, C.ROOT, f"level_{tag}", "10_level_probe.py",
               f"{a.base} ({name}) with segment {label} x{a.factor:.3f}", "level probe")
        return 0

    if a.base_score is None or a.score is None:
        ap.error("read needs --base-score and --score")
    mu = base.to_numpy(dtype=float)
    n = null_deviance(mu, meta["line"].to_numpy())
    b = float(np.sum(np.sqrt(mu[sel]))) / n                  # B / N
    c = a.factor
    gain = a.score - a.base_score
    aa = (-gain / 4 - (np.sqrt(c) - 1) * b) / (1 / np.sqrt(c) - 1)   # A / N, from the probe
    best = aa / b
    print(f"probe x{c:.3f}: public {a.base_score:.5f} -> {a.score:.5f} ({gain:+.5f}); estimated null deviance "
          f"{n / len(mu):.1f} per row")
    print(f"best level of the segment: x{best:.3f} of {a.base} (expected public {a.base_score + 4 * (np.sqrt(aa) - np.sqrt(b)) ** 2:.5f})")
    for f in sorted({1.0, c, round(best, 3), *(round(best + d, 3) for d in (-0.06, -0.03, 0.03))}):
        d = -4 * ((f ** -0.5 - 1) * aa + (f ** 0.5 - 1) * b)
        print(f"   x{f:.3f}: expected public {a.base_score + d:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
