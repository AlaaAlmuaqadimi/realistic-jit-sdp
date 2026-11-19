"""Central configuration: every constant that affects results lives here.

All rules (features, exclusion rule, protocol parameters) are fixed *before*
modeling, as required by the proposal (Sections 8.2-8.4).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "apachejit_total.csv"
DATA_URL = "https://raw.githubusercontent.com/hosseinkshvrz/apachejit/master/dataset/apachejit_total.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS_DIR = ROOT / "models"

# Second dataset: Kamei et al. (2013), six projects, as distributed in the
# replication package of Yang et al. (2016) / Fu & Menzies (2017).
KAMEI_DIR = ROOT / "data" / "kamei"
KAMEI_URL = "https://github.com/WeiFoo/RevisitUnsupervised/tree/master/jit/input"
KAMEI_PROJECTS = ["bugzilla", "columba", "jdt", "mozilla", "platform", "postgres"]
DATASET = "apachejit"          # set by run_all.py --dataset


def use_dataset(name: str) -> None:
    """Point all output folders at the chosen dataset (ApacheJIT keeps the
    original locations so earlier results stay valid)."""
    global DATASET, RESULTS, FIGURES, MODELS_DIR
    DATASET = name
    suffix = "" if name == "apachejit" else f"_{name}"
    RESULTS = ROOT / f"results{suffix}"
    FIGURES = RESULTS / "figures"
    MODELS_DIR = ROOT / f"models{suffix}"

SEED = 42

# ---------------------------------------------------------------- features
LABEL = "buggy"
TIME = "author_date"           # unix seconds, commit author date
PROJECT = "project"
ID = "commit_id"

# Kamei et al. (2013) style metrics available at commit time.
FEATURES = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc",
            "aexp", "arexp", "asexp", "fix"]
# Heavily skewed count-like features -> log1p (stateless transform).
LOG_FEATURES = ["la", "ld", "nf", "nd", "ns", "ndev", "age", "nuc",
                "aexp", "arexp", "asexp"]
# Fields that must never be used as features (identifiers, time, label).
FORBIDDEN = ["commit_id", "project", "buggy", "year", "author_date", "date"]
CORR_THRESHOLD = 0.80          # Spearman |rho| for the LR collinearity filter

# ---------------------------------------------------------------- protocols
TEST_FRAC = 0.30               # P1/P2/P3: 70/30
VAL_FRAC = 0.20                # last 20% of training -> threshold selection
GAPS = [30, 90, 180]           # P3 latency gaps (days); 90 is the default
DEFAULT_GAP = 90
P1_SEEDS = [0, 1, 2]           # repeated random splits for P1

# Exclusion rule (fixed before modeling): a project is retained only if the
# chronological 70/30 split has >= MIN_CLASS commits of each class in both
# the training and the test part.
MIN_CLASS = 50

# Rolling windows: train 12 months, test the next 3 months, step 3 months.
# The P3 latency gap is also applied between the training and test window.
WIN_TRAIN_MONTHS = 12
WIN_TEST_MONTHS = 3
WIN_STEP_MONTHS = 3
WIN_MIN_POS_TRAIN = 30
WIN_MIN_POS_TEST = 10
WIN_MIN_NEG_TEST = 10

# ---------------------------------------------------------------- models
MODELS = ["LR", "RF", "XGB"]
BASELINES = ["Prevalence", "LT"]      # LT = inverse size (Yang et al., 2016)
STRATEGIES = ["none", "weight", "under"]
DEFAULT_STRATEGY = "weight"
TUNING_FOLDS = 3

# ---------------------------------------------------------------- metrics
EFFORT_BUDGET = 0.20
SHAP_SAMPLE = 400              # test commits per window for SHAP
ALPHA = 0.05
