"""Figures and tables for the write-up.

Everything is read from the pipeline's own reports and registers, so the write-up shows the
numbers the pipeline wrote. Colors come from a colorblind-checked palette (blue, orange, aqua:
every pair passes the CVD and normal-vision checks; aqua is below 3:1 contrast, so each series
also has its own marker shape and the numbers are in a table).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
REPO = FINAL.parent
R1, R2 = REPO / "1st round", REPO / "2nd round"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# Week 1 models in 1st round/reports/06_geo/scores.csv, weakest first.
WEEK1_MODELS = [
    ("glm", "GLM"),
    ("peril", "one GBM per peril"),
    ("total", "single LightGBM (baseline)"),
    ("fs_gamma", "frequency × Gamma claim size"),
    ("fs_lognorm", "frequency × lognormal claim size"),
    ("fs_peril", "frequency × claim size, per peril"),
    ("geo", "geometric mean of the six"),
    ("recalibrated geo (fit on the other year)", "geometric mean, recalibrated"),
]

# Week 2 approaches in 2nd round/reports/05_model/holdout.csv: (key, label, color, marker).
WEEK2_APPROACHES = [
    ("transfer LightGBM", "LightGBM on the developed prior", BLUE, "o"),
    ("developed prior alone", "developed prior alone", ORANGE, "D"),
    ("LightGBM without prior", "LightGBM on Week 2 data only", AQUA, "s"),
]
WEEK2_LINES = [("all lines", "All lines"), ("motor", "Motor"), ("buildings", "Buildings"), ("contents", "Contents")]

SEGMENT_NAMES = {
    "motor_new_m": "Motor, new business, men",
    "motor_new_f": "Motor, new business, women",
    "motor_renewal_m": "Motor, renewals, men",
    "motor_renewal_f": "Motor, renewals, women",
    "buildings_new": "Buildings, new business",
    "buildings_renewal": "Buildings, renewals",
    "contents_new_f": "Contents, new business, women",
    "contents_new_m": "Contents, new business, men",
    "contents_renewal_f": "Contents, renewals, women",
    "contents_renewal_m": "Contents, renewals, men",
}


def use_style(scale: float = 1.0) -> None:
    """Quiet chart chrome: hairline grid, muted axes, system sans."""
    mpl.rcParams.update({
        "font.family": ["Segoe UI", "Calibri", "DejaVu Sans"],
        "font.size": 8.5 * scale,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9.5 * scale,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8 * scale,
        "lines.solid_capstyle": "round",
    })


# ---------------------------------------------------------------------------------------- data

def week1_backtests() -> pd.DataFrame:
    """Pooled back-test D² of every Week 1 model, one column per scored year."""
    s = pd.read_csv(R1 / "reports" / "06_geo" / "scores.csv")
    d = s.loc[s["line"] == "all lines"].pivot_table(index="model", columns="year", values="d2")
    return d.loc[[k for k, _ in WEEK1_MODELS]]


def week2_transfer() -> pd.DataFrame:
    """Forward-test D² (fit 2023, score 2024) per line for the three Week 2 approaches."""
    h = pd.read_csv(R2 / "reports" / "05_model" / "holdout.csv")
    d = h.pivot_table(index="line", columns="model", values="d2")
    return d.loc[[k for k, _ in WEEK2_LINES], [k for k, *_ in WEEK2_APPROACHES]]


def week2_register() -> pd.DataFrame:
    return pd.read_csv(R2 / "submissions" / "versions.csv")


def segment_levels() -> pd.DataFrame:
    lv = pd.read_csv(R2 / "reports" / "11_joint" / "levels.csv")
    lv["name"] = lv["segment"].map(SEGMENT_NAMES)
    return lv.sort_values("share_of_total", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------------------------- figures

def fig_week1_models(size=(6.6, 2.5), scale: float = 1.0):
    """Every Week 1 model on both back-test years; the gap between the dots is the year-to-year change."""
    use_style(scale)
    d = week1_backtests()
    y = list(range(len(d)))
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.hlines(y, d[2023], d[2024], color=AXIS, lw=1.2, zorder=1)
    ax.scatter(d[2023], y, s=40 * scale, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="scored on 2023 (fitted on 2021-22)")
    ax.scatter(d[2024], y, s=36 * scale, marker="D", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="scored on 2024 (fitted on 2021-23)")
    ax.set_yticks(y, [label for _, label in WEEK1_MODELS])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Back-test D², all lines (0 = predicting the average for everyone)")
    ax.set_title("Week 1: every model scored on a year it had not seen")
    fig.legend(loc="outside lower center", ncols=2, handletextpad=0.3, columnspacing=2.0)
    return fig


def fig_week2_transfer(size=(6.6, 2.2), scale: float = 1.0):
    """The three Week 2 approaches on the forward test, per line."""
    use_style(scale)
    d = week2_transfer()
    y = list(range(len(d)))[::-1]
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    for (key, label, color, marker), dy in zip(WEEK2_APPROACHES, (0.2, 0.0, -0.2)):   # dodge close values
        ax.scatter(d[key], [v + dy for v in y], s=40 * scale, marker=marker, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=label)
    ax.set_yticks(y, [label for _, label in WEEK2_LINES])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, None)
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_xlabel("Forward-test D²: fitted on 2023, scored on 2024")
    ax.set_title("Week 2: the developed-market prior against two years of local data")
    fig.legend(loc="outside lower center", ncols=3, handletextpad=0.3, columnspacing=1.6)
    return fig


def fig_levels(size=(6.6, 3.2), scale: float = 1.0):
    """Best level per segment (factor on v5): for the public rows and for rows not yet scored."""
    use_style(scale)
    lv = segment_levels()
    y = list(range(len(lv)))[::-1]
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.axvline(1.0, color=AXIS, lw=1.2, zorder=0)
    ax.hlines(y, lv["best_unseen"] - lv["sd_unseen"], lv["best_unseen"] + lv["sd_unseen"],
              color=BLUE, lw=2, alpha=0.55, zorder=1)
    ax.scatter(lv["best_unseen"], y, s=46 * scale, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="best for rows not yet scored, ±1 sd (≈ v12)")
    ax.scatter(lv["best_public"], y, s=40 * scale, marker="D", color=ORANGE, edgecolor=SURFACE,
               linewidth=1.5, zorder=2, label="best for the public rows (≈ v13)")
    ax.set_yticks(y, [f"{n}  ({s:.0%})" for n, s in zip(lv["name"], lv["share_of_total"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Factor on v5's segment total (1 = v5)")
    ax.set_title("Best level per segment, as a factor on v5 (share of 2025 cost in brackets)")
    fig.legend(loc="outside lower center", ncols=2, handletextpad=0.3, columnspacing=2.0)
    return fig


# --------------------------------------------------------------------------------------- tables

# One line per Week 2 version; the full descriptions are in 2nd round/submissions/versions.csv.
VERSION_NOTES = {
    "v1": "starter: one LightGBM per line, no prior, no trend or large-loss treatment",
    "v2": "LightGBM on the developed prior, with new-business and gender factors",
    "v3": "ensemble on developed priors, recalibration slope fitted (kept for comparison)",
    "v4": "ensemble on developed priors, level only (slope 1)",
    "v5": "earlier best file's segment totals; ranking 50% that file, 50% ensemble",
    "v6": "as v5, ranking 25% / 75%",
    "v7": "probe: v5 with motor new business x1.15",
    "v8": "probe: v5 with buildings new business x1.08",
    "v9": "probe: v5 with contents new business x1.10",
    "v10": "probe: v5 with motor renewals x1.07",
    "v11": "v5 with every segment at its best level from all scores",
    "v12": "v5 with every segment at its best level for rows not yet scored",
    "v13": "v5 with every segment at its best level for the public rows (Week 2 file)",
    "v14": "v12's levels; buildings and contents ranked without this-year weather; motor spread x1.05",
    "v15": "v12 with the motor spread x1.05 only",
}


def table_register() -> pd.DataFrame:
    reg = week2_register()
    return pd.DataFrame({
        "Version": reg["version"],
        "Public D²": reg["public_score"].map(lambda s: "not scored" if pd.isna(s) else f"{s:.5f}"),
        "What it is": reg["version"].map(VERSION_NOTES).fillna(reg["description"]),
    })


def table_week2_transfer() -> pd.DataFrame:
    d = week2_transfer()
    out = pd.DataFrame({"Line": [label for _, label in WEEK2_LINES]})
    for key, label, *_ in WEEK2_APPROACHES:
        out[label] = d[key].map("{:.4f}".format).to_numpy()
    return out


def table_levels() -> pd.DataFrame:
    lv = segment_levels()
    return pd.DataFrame({
        "Segment": lv["name"],
        "Share of 2025 cost": lv["share_of_total"].map("{:.1%}".format),
        "Public-row luck sd": lv["public_luck_sd"].map("{:.1%}".format),
        "Best for public rows": lv["best_public"].map("{:.3f}".format),
        "Best for unseen rows": lv["best_unseen"].map("{:.3f}".format),
        "sd": lv["sd_unseen"].map("{:.3f}".format),
    })


def markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


if __name__ == "__main__":  # quick preview: python final/src/final_figures.py <folder>
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else FINAL / "reproduced"
    out.mkdir(parents=True, exist_ok=True)
    fig_week1_models().savefig(out / "week1_models.png", dpi=200)
    fig_week2_transfer().savefig(out / "week2_transfer.png", dpi=200)
    fig_levels().savefig(out / "levels.png", dpi=200)
    print(week1_backtests().round(4))
    print(markdown(table_week2_transfer()))
    print(f"previews in {out}")
