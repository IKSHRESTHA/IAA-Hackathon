"""Guardrails: submission integrity, reasonableness checks, escalation and run manifest.

Each check returns a :class:`Check` record. ``severity="block"`` means the
submission must not be produced; ``severity="escalate"`` means it may be produced
but needs human (actuarial) sign-off before it is used; ``"info"`` is recorded only.
The thresholds live in ``configs/*.yaml`` (section ``guardrails``).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config


# Every threshold used below, with its default. The YAML ``guardrails:`` section overrides
# these; unknown keys raise, so a mistyped threshold can never be silently ignored.
GUARDRAIL_DEFAULTS: dict[str, float] = {
    "max_abs_loss_ratio_shift": 0.25,      # |implied 2025 LR / developed last-year LR - 1|
    "max_pp_shift_vs_history": 0.15,       # |2025 predicted PP / developed history PP - 1|
    "max_prediction_to_median_ratio": 200,  # max / median predicted rate within a line
    "psi_warning": 0.10,
    "psi_escalate": 0.25,
    "min_validation_d2": 0.0,              # holdout D^2 at or below this blocks
    "max_line_ae_deviation": 0.05,         # holdout A/E outside 1 +/- this escalates
}


def guardrail_settings(cfg: Config) -> dict[str, float]:
    """Thresholds from the YAML ``guardrails:`` section merged over the defaults."""
    g = cfg.section("guardrails")
    unknown = set(g) - set(GUARDRAIL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown guardrails config keys: {sorted(unknown)}")
    return {k: float(g.get(k, v)) for k, v in GUARDRAIL_DEFAULTS.items()}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    severity: str          # block | escalate | info
    detail: str
    value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_submission(sub: pd.DataFrame, expected_row_ids: Iterable[str]) -> list[Check]:
    """Hard integrity checks: any failure blocks the submission."""
    expected = pd.Index(pd.Series(list(expected_row_ids), dtype="string"))
    checks = [Check("columns_exact", list(sub.columns) == ["row_id", "prediction"], "block",
                    f"columns={list(sub.columns)}")]
    if not checks[0].passed:
        return checks
    ids = pd.Index(sub["row_id"].astype("string"))
    pred = pd.to_numeric(sub["prediction"], errors="coerce").to_numpy(dtype=float)
    n_dup = int(ids.duplicated().sum())
    missing = expected.difference(ids)
    extra = ids.difference(expected)
    checks += [
        Check("row_ids_unique", n_dup == 0, "block", f"{n_dup} duplicated row_ids", n_dup),
        Check("row_ids_complete", len(missing) == 0, "block",
              f"{len(missing)} expected row_ids missing", len(missing)),
        Check("row_ids_no_extra", len(extra) == 0, "block", f"{len(extra)} unexpected row_ids", len(extra)),
        Check("predictions_finite", bool(np.isfinite(pred).all()), "block",
              f"{int((~np.isfinite(pred)).sum())} non-finite predictions"),
        Check("predictions_non_negative", bool((pred >= 0).all()), "block",
              f"{int((pred < 0).sum())} negative predictions"),
        # The Tweedie deviance is infinite for a zero prediction on a row with a claim.
        Check("predictions_strictly_positive", bool((pred > 0).all()), "block",
              f"{int((pred <= 0).sum())} predictions <= 0"),
    ]
    return checks


def reasonableness_checks(
    pred: pd.DataFrame,
    history: pd.DataFrame,
    cfg: Config,
) -> list[Check]:
    """Portfolio-level checks of the 2025 predictions against developed history.

    ``pred``: row_id, line, exposure, prediction, written_premium.
    ``history``: line, year, exposure, developed_cost (2025 cost level), earned_premium.
    """
    g = guardrail_settings(cfg)
    out: list[Check] = []
    max_lr_shift = g["max_abs_loss_ratio_shift"]
    max_ratio = g["max_prediction_to_median_ratio"]
    max_pp_shift = g["max_pp_shift_vs_history"]
    for line, p in pred.groupby("line"):
        h = history.loc[history["line"] == line]
        pp_pred = p["prediction"].sum() / p["exposure"].sum()
        pp_hist = h["developed_cost"].sum() / h["exposure"].sum()
        rel = pp_pred / pp_hist - 1.0
        out.append(Check(f"{line}:pure_premium_vs_history", abs(rel) <= max_pp_shift, "escalate",
                         f"2025 predicted PP {pp_pred:,.1f} vs developed 2021-24 PP at 2025 level "
                         f"{pp_hist:,.1f} ({rel:+.1%})", float(rel)))
        lr_pred = p["prediction"].sum() / (p["written_premium"] * p["exposure"]).sum()
        last = h.loc[h["year"] == h["year"].max()]
        lr_last = last["developed_cost"].sum() / last["earned_premium"].sum()
        shift = lr_pred / lr_last - 1.0
        out.append(Check(f"{line}:loss_ratio_shift", abs(shift) <= max_lr_shift, "escalate",
                         f"implied 2025 LR {lr_pred:.1%} vs developed {int(last['year'].max())} LR "
                         f"(2025 cost level) {lr_last:.1%}", float(shift)))
        rate = p["prediction"] / p["exposure"]
        ratio = float(rate.max() / rate.median())
        out.append(Check(f"{line}:max_rate_to_median", ratio <= max_ratio, "escalate",
                         f"max/median predicted rate = {ratio:,.1f}", ratio))
        pv = p["prediction"].to_numpy(dtype=float)
        ok = bool(np.isfinite(pv).all() and (pv > 0).all())
        out.append(Check(f"{line}:min_prediction_positive", ok, "block",
                         f"min prediction {np.nanmin(pv):.4f}; non-finite {int((~np.isfinite(pv)).sum())}",
                         float(np.nanmin(pv))))
    return out


def validation_checks(backtests: pd.DataFrame, cfg: Config) -> list[Check]:
    """Back-test results gate: D^2 above the baseline and A/E within tolerance."""
    g = guardrail_settings(cfg)
    min_d2, tol = g["min_validation_d2"], g["max_line_ae_deviation"]
    out = []
    for r in backtests.itertuples(index=False):
        tag = f"{r.line}:{r.holdout_year}:{r.model}"
        out.append(Check(f"{tag}:d2_above_baseline", r.d2 > min_d2, "block",
                         f"D2={r.d2:.4f} (min {min_d2})", float(r.d2)))
        out.append(Check(f"{tag}:ae_within_tolerance", abs(r.ae - 1) <= tol, "escalate",
                         f"A/E={r.ae:.3f} (tolerance +/-{tol:.0%})", float(r.ae)))
    return out


def drift_checks(psi: pd.DataFrame, cfg: Config) -> list[Check]:
    """PSI of model features between the last training year and the prediction year."""
    g = guardrail_settings(cfg)
    warn, esc = g["psi_warning"], g["psi_escalate"]
    out = []
    for r in psi.itertuples(index=False):
        if r.psi >= esc:
            out.append(Check(f"{r.line}:psi:{r.feature}", False, "escalate",
                             f"PSI={r.psi:.3f} >= {esc}", float(r.psi)))
        elif r.psi >= warn:
            out.append(Check(f"{r.line}:psi:{r.feature}", True, "info",
                             f"PSI={r.psi:.3f} (warning band)", float(r.psi)))
    return out


def checks_frame(checks: Iterable[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.as_dict() for c in checks],
                        columns=["name", "passed", "severity", "detail", "value"])


def blocking_failures(checks: pd.DataFrame) -> pd.DataFrame:
    return checks.loc[(checks["severity"] == "block") & ~checks["passed"].astype(bool)]


def escalations(checks: pd.DataFrame) -> pd.DataFrame:
    return checks.loc[(checks["severity"] == "escalate") & ~checks["passed"].astype(bool)]


ESCALATION_COLUMNS = ["name", "severity", "detail", "value", "signed_off_by", "decision", "date"]


def escalation_register(checks: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """Open escalations with empty sign-off fields, for the accountable actuary to complete.

    If a ``previous`` register is given, existing sign-offs for the same check name are
    carried over (so re-running the pipeline does not erase recorded decisions).
    """
    esc = escalations(checks)[["name", "severity", "detail", "value"]].copy()
    for c in ("signed_off_by", "decision", "date"):
        esc[c] = ""
    if previous is not None and not previous.empty and "name" in previous:
        prev = previous.set_index("name")
        for c in ("signed_off_by", "decision", "date"):
            if c in prev:
                carried = esc["name"].map(prev[c].fillna("").astype(str))
                esc[c] = carried.where(carried.notna(), esc[c]).fillna("")
    return esc[ESCALATION_COLUMNS].reset_index(drop=True)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def artifact_hashes(paths: Iterable[Path], root: Path | None = None) -> dict[str, str]:
    """SHA-256 of each existing file (keys relative to ``root`` when given)."""
    out: dict[str, str] = {}
    for p in sorted(Path(x) for x in paths):
        if p.is_file():
            key = str(p.relative_to(root)) if root is not None and p.is_relative_to(root) else str(p)
            out[key] = file_sha256(p)
    return out


# --------------------------------------------------------------------------- clean-tree gate
SOURCE_PREFIXES: tuple[str, ...] = ("src/", "scripts/", "configs/")
_UNTRACKED_BLOCKING_PREFIXES: tuple[str, ...] = ("src/", "configs/")


def uncommitted_source_changes(porcelain: str, prefixes: tuple[str, ...] = SOURCE_PREFIXES) -> list[str]:
    """Paths that make a run irreproducible from the current commit.

    Parses ``git status --porcelain --untracked-files=all``. Any change to a *tracked*
    file under ``prefixes`` counts (modified, added, deleted, renamed, ...). Untracked
    files count only under ``src/`` and ``configs/``, because only those can change what
    the pipeline imports or reads (an untracked analysis script under ``scripts/`` does not).
    """
    out: list[str] = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:                      # rename: "R  old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if status == "!!" or not path.startswith(prefixes) or "__pycache__/" in path:
            continue
        if status == "??" and not path.startswith(_UNTRACKED_BLOCKING_PREFIXES):
            continue
        out.append(path)
    return out


def git_porcelain(repo_root: Path | None = None) -> str:
    """``git status --porcelain --untracked-files=all`` for the repository (raises if git fails)."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                          capture_output=True, text=True, check=True, cwd=root).stdout


# --------------------------------------------------------------------------- manifest
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:  # noqa: BLE001 - manifest must never fail the run
        return "unavailable"


def run_manifest(cfg: Config, data_hashes: dict[str, str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to reproduce (or audit) a run."""
    pkgs = {}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "glum", "pyyaml"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "not installed"
    return {
        "run_name": cfg.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_path": str(cfg.source_path),
        "config_fingerprint": cfg.fingerprint(),
        "seed": cfg.seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "data_sha256": data_hashes,
        **(extra or {}),
    }


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
