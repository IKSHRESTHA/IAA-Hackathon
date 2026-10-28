"""Figures and tables for the write-up and the video slides.

Everything is read from the repository's own registers and reports, so the documents show the
same numbers the pipeline wrote. Colors follow a colorblind-checked palette (blue / orange,
validated for adjacent-pair CVD separation and 3:1 contrast on the light surface).
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
BLUE, ORANGE = "#2a78d6", "#eb6834"

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

# Short tags for the scored Week 2 files, in the order they were made.
STEP_TAGS = {
    "v1": "starter\nLightGBM",
    "v4": "ensemble\non priors",
    "v5": "best levels,\nnew ranking",
    "v8": "probe:\nbuildings",
    "v10": "probe:\nmotor ren.",
    "v11": "joint\nlevels",
    "v12": "levels for\nunseen rows",
    "v13": "levels for\npublic rows",
    "v14": "ranking\ntweaks",
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

def week2_register() -> pd.DataFrame:
    return pd.read_csv(R2 / "submissions" / "versions.csv")


def week2_scored() -> pd.DataFrame:
    reg = week2_register().dropna(subset=["public_score"]).copy()
    reg["n"] = reg["version"].str[1:].astype(int)
    return reg.sort_values("n")


def segment_levels() -> pd.DataFrame:
    lv = pd.read_csv(R2 / "reports" / "11_joint" / "levels.csv")
    lv["name"] = lv["segment"].map(SEGMENT_NAMES)
    return lv.sort_values("share_of_total", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------------------------- figures

def fig_journey(size=(6.6, 2.55), scale: float = 1.0):
    """Public score of every scored Week 2 file, in the order made; the two final picks in blue."""
    use_style(scale)
    df = week2_scored()
    x = range(len(df))
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.plot(list(x), df["public_score"], color=AXIS, lw=1.6, zorder=1)
    picks = df["version"].isin(["v12", "v13"]).to_numpy()
    ax.scatter([i for i, p in zip(x, picks) if not p], df.loc[~picks, "public_score"], s=42 * scale,
               color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.scatter([i for i, p in zip(x, picks) if p], df.loc[picks, "public_score"], s=58 * scale,
               color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    labels = {"v1": (0, -13), "v4": (0, -13), "v5": (0, 8), "v12": (-4, -14), "v13": (0, 8),
              "v14": (0, -14)}
    for i, (v, s) in enumerate(zip(df["version"], df["public_score"])):
        if v in labels:
            dx, dy = labels[v]
            ax.annotate(f"{s:.5f}", (i, s), xytext=(dx, dy), textcoords="offset points", ha="center",
                        fontsize=7.6 * scale, color=INK if v in ("v12", "v13") else INK2,
                        fontweight="bold" if v == "v13" else "normal")
    ax.set_xticks(list(x), [f"{v}\n{STEP_TAGS.get(v, '')}" for v in df["version"]], fontsize=7.2 * scale)
    ax.set_ylim(0.118, 0.178)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Public D² (Tweedie, p = 1.5)")
    ax.set_title("Week 2 public score of every scored file, in the order made (blue: the two final picks)")
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


if __name__ == "__main__":  # quick preview: python final/src/final_figures.py
    out = FINAL / "src" / "_preview"
    out.mkdir(exist_ok=True)
    fig_journey().savefig(out / "journey.png", dpi=200)
    fig_levels().savefig(out / "levels.png", dpi=200)
    print(markdown(table_levels()))
    print(f"previews in {out}")
