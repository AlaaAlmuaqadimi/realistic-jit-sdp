"""Loading, validation (dataset-readiness gate) and the project exclusion rule."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from . import config as C


def _data_files():
    if C.DATASET == "kamei":
        return [C.KAMEI_DIR / f"{p}.csv" for p in C.KAMEI_PROJECTS]
    return [C.DATA_FILE]


def file_sha256(paths=None) -> str:
    """SHA-256 of the dataset file(s) (concatenated in a fixed order)."""
    h = hashlib.sha256()
    for path in paths or _data_files():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def load() -> pd.DataFrame:
    """Load the active dataset in the common schema used by the pipeline."""
    if C.DATASET == "kamei":
        return _load_kamei()
    if not C.DATA_FILE.exists():
        raise FileNotFoundError(
            f"{C.DATA_FILE} not found. Download it from {C.DATA_URL}")
    df = pd.read_csv(C.DATA_FILE)
    df[C.LABEL] = df[C.LABEL].astype(bool).astype(int)
    df["fix"] = df["fix"].astype(bool).astype(int)
    df["date"] = pd.to_datetime(df[C.TIME], unit="s")
    df = df.sort_values([C.TIME, C.ID], kind="mergesort").reset_index(drop=True)
    return df


# Kamei et al. (2013) column names -> common schema
_KAMEI_MAP = {"nm": "nd", "entropy": "ent", "pd": "age", "npt": "nuc",
              "exp": "aexp", "rexp": "arexp", "sexp": "asexp", "bug": C.LABEL}


def _load_kamei() -> pd.DataFrame:
    """In this dataset LA and LD are normalised by LT (lines in the touched
    files before the change), so absolute churn = (la + ld) * lt is used as
    the review-effort proxy, as in Yang et al. (2016). Model features keep
    the original (normalised) values."""
    frames = []
    for p in C.KAMEI_PROJECTS:
        f = C.KAMEI_DIR / f"{p}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Download it from {C.KAMEI_URL}")
        d = pd.read_csv(f).rename(columns=_KAMEI_MAP)
        d["date"] = pd.to_datetime(d["commitdate"], format="%Y/%m/%d %H:%M")
        # resolution-independent conversion to unix seconds
        d[C.TIME] = ((d["date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
        d[C.PROJECT] = f"kamei/{p}"
        d[C.ID] = p + "-" + d["transactionid"].astype(str)
        d["effort"] = np.maximum((d["la"] + d["ld"]) * d["lt"], 1.0)
        d[C.LABEL] = d[C.LABEL].astype(int)
        d["fix"] = d["fix"].astype(int)
        frames.append(d[[C.ID, C.PROJECT, C.LABEL, C.TIME, "date", "effort", "lt", *C.FEATURES]])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values([C.TIME, C.ID], kind="mergesort").reset_index(drop=True)


def effort(df: pd.DataFrame) -> np.ndarray:
    """Review effort proxy = lines added + deleted (min 1 line). Datasets
    with normalised churn provide an explicit 'effort' column."""
    if "effort" in df:
        return df["effort"].to_numpy(dtype=float)
    return np.maximum(df["la"].to_numpy() + df["ld"].to_numpy(), 1).astype(float)


# --------------------------------------------------------------- validation
def validate(df: pd.DataFrame) -> dict:
    """Schema/type/missing/duplicate checks plus the readiness gate (Sec 8.2)."""
    out: dict = {}
    required = [C.ID, C.PROJECT, C.LABEL, C.TIME, *C.FEATURES]
    out["missing_columns"] = [c for c in required if c not in df.columns]
    out["n_commits"] = int(len(df))
    out["n_projects"] = int(df[C.PROJECT].nunique())
    out["missing_values"] = int(df[required].isna().sum().sum())
    out["duplicate_commit_ids"] = int(df.duplicated(C.ID).sum())
    out["duplicate_feature_rows"] = int(df.duplicated([C.PROJECT, *C.FEATURES, C.TIME]).sum())
    num = [f for f in C.FEATURES]
    out["negative_values"] = {f: int((df[f] < 0).sum()) for f in num if (df[f] < 0).any()}
    out["date_range"] = (str(df["date"].min()), str(df["date"].max()))
    out["prevalence"] = float(df[C.LABEL].mean())
    out["has_fix_timestamps"] = False  # ApacheJIT does not link bug-fix dates
    out["sha256"] = file_sha256()

    per_project = (df.groupby(C.PROJECT)
                   .agg(commits=(C.LABEL, "size"), buggy=(C.LABEL, "sum"),
                        prevalence=(C.LABEL, "mean"),
                        first=("date", "min"), last=("date", "max"))
                   .reset_index())
    per_year = (df.assign(year=df["date"].dt.year)
                .groupby("year").agg(commits=(C.LABEL, "size"),
                                     prevalence=(C.LABEL, "mean")).reset_index())
    gate = {
        "1_timestamp": df[C.TIME].notna().all(),
        "2_change_metrics": not out["missing_columns"],
        "3_labels_documented": set(df[C.LABEL].unique()) <= {0, 1},
        "4_enough_pos_neg": bool((per_project["buggy"] >= C.MIN_CLASS).sum()
                                 >= min(5, len(per_project))),
        "5_latency_protocol": True,  # gap approximation (no fix timestamps)
    }
    out["readiness_gate"] = {k: bool(v) for k, v in gate.items()}
    out["readiness_passed"] = all(gate.values())
    return {"summary": out, "per_project": per_project, "per_year": per_year}


def chrono_split_index(n: int, test_frac: float = C.TEST_FRAC) -> int:
    """Index of the first test commit for a chronological split."""
    return n - int(math.ceil(n * test_frac))


def exclusion_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered exclusion rule to every project."""
    rows = []
    for proj, g in df.groupby(C.PROJECT):
        g = g.sort_values(C.TIME, kind="mergesort")
        k = chrono_split_index(len(g))
        tr, te = g[C.LABEL].iloc[:k], g[C.LABEL].iloc[k:]
        counts = dict(train_pos=int(tr.sum()), train_neg=int((1 - tr).sum()),
                      test_pos=int(te.sum()), test_neg=int((1 - te).sum()))
        keep = min(counts.values()) >= C.MIN_CLASS
        reason = "" if keep else "fewer than %d commits of a class in a split" % C.MIN_CLASS
        rows.append(dict(project=proj, commits=len(g), **counts, retained=keep, reason=reason))
    return pd.DataFrame(rows)
