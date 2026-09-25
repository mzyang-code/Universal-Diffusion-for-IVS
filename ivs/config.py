#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2026

@author: mzyang

Configuration loader: every constant is read from the experiment json selected by
--config or $IVS_CONFIG.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Config discovery
def _resolve_config_path() -> Path:
    """--config <path> on argv, else $IVS_CONFIG, else raise."""
    argv = sys.argv
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        if tok.startswith("--config="):
            return Path(tok.split("=", 1)[1]).expanduser()
    env = os.environ.get("IVS_CONFIG")
    if env:
        return Path(env).expanduser()
    raise RuntimeError(
        "No experiment config selected. Pass --config configs/<experiment>.json "
        "or export IVS_CONFIG=configs/<experiment>.json. Available configs: "
        + ", ".join(sorted(p.name for p in (REPO_ROOT / "configs").glob("*.json")))
    )


_CONFIG_PATH = _resolve_config_path()
if not _CONFIG_PATH.is_absolute():
    _CONFIG_PATH = (REPO_ROOT / _CONFIG_PATH).resolve()
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found: {_CONFIG_PATH}")
with open(_CONFIG_PATH, "r") as _fp:
    CONFIG: dict = json.load(_fp)

CONFIG_PATH = _CONFIG_PATH
EXPERIMENT = str(CONFIG.get("experiment", _CONFIG_PATH.stem))
FAMILY = str(CONFIG.get("family", "diffusion"))          # "diffusion" | "volgan"


# Section access helpers (tolerant: a missing key yields None, not a KeyError)
def _sec(name: str) -> dict:
    return CONFIG.get(name, {}) or {}


def _get(section: dict, key: str, cast=None, default=None):
    if key not in section:
        return default
    val = section[key]
    return val if cast is None else cast(val)


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(raw: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-fallback}`` against the environment."""
    def sub(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
    return _PLACEHOLDER.sub(sub, raw)


def _path(section: dict, key: str, default: Any = None) -> Path | None:
    """Resolve a ``paths.*`` entry: expand placeholders, anchor at REPO_ROOT."""
    raw = section.get(key, default)
    if raw is None:
        return None
    p = Path(_expand(str(raw))).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# Paths
_pa = _sec("paths")
# Data (read-only inputs produced by the data-preparation stage).
TENSOR_DIR = _path(_pa, "tensor_dir", "data/tensors")
CLEAN_DIR = _path(_pa, "clean_dir", "data/cleaning")
RAW_DIR = _path(_pa, "raw_dir", "data/raw")

# Run artifacts. ARTIFACT_DIR is the root the heavy directories derive from; each
# may be overridden, which is how the OOS configs reuse the training checkpoint
# and normalizer while writing samples to a separate tree.
ARTIFACT_DIR = _path(_pa, "artifact_dir", f"artifacts/{EXPERIMENT}")
CKPT_DIR = _path(_pa, "ckpt_dir") or (ARTIFACT_DIR / "checkpoints")
NORM_DIR = _path(_pa, "norm_dir") or (ARTIFACT_DIR / "normalizer")
SAMPLE_DIR = _path(_pa, "sample_dir") or (ARTIFACT_DIR / "samples")
LOG_DIR = _path(_pa, "log_dir") or (ARTIFACT_DIR / "logs")
EVAL_OUTPUT_DIR = _path(_pa, "eval_output_dir") or (ARTIFACT_DIR / "evaluate_outputs")

BANDWIDTHS_JSON = TENSOR_DIR / "_bandwidths.json"
UNIVERSE_JSON = TENSOR_DIR / "_universe.json"

for _p in (CKPT_DIR, SAMPLE_DIR, LOG_DIR, NORM_DIR, EVAL_OUTPUT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Reproducibility
GLOBAL_SEED: int = int(_sec("reproducibility").get("global_seed", 20260611))


def set_global_seed(seed: int | None = None, deterministic: bool = True) -> int:
    """Seed random, numpy and torch (CPU + all CUDA devices); return the seed used.
    deterministic also forces cuDNN into deterministic mode. TF32 matmul stays on:
    it is deterministic for fixed inputs, so it does not conflict."""
    s = int(GLOBAL_SEED if seed is None else seed)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass
    return s


# Universe
TICKERS: tuple[str, ...] = tuple(CONFIG["tickers"])
VERSIONS: tuple[str, ...] = tuple(CONFIG["versions"].keys())

_uni = _sec("universe")
N_CANDIDATES = _get(_uni, "n_candidates", int)
N_SELECT = _get(_uni, "n_select", int)
SELECT_SEED = _get(_uni, "select_seed", int)
UNIVERSE_MIN_VALID_DAYS = _get(_uni, "min_valid_days", int)

# Data-cleaning gates (options_scrubber.py): quote quality only -- price, spread,
# liquidity, delta range, maturity floor, Vega minimum. Grid/range concerns are
# left to the grid calibration and the surface build downstream.
_dc = _sec("data_cleaning")
MIN_MID_PRICE = _get(_dc, "min_mid_price", float)
IV_LOW = _get(_dc, "iv_low", float)
IV_HIGH = _get(_dc, "iv_high", float)
MAX_IV_SPREAD = _get(_dc, "max_iv_spread", float)
MID_BREAKPOINT = _get(_dc, "mid_breakpoint", float)
REL_SPREAD_MAX_HIGH_MID = _get(_dc, "rel_spread_max_high_mid", float)
ABS_SPREAD_MAX_LOW_MID = _get(_dc, "abs_spread_max_low_mid", float)
MIN_OPEN_INTEREST = _get(_dc, "min_open_interest", int)
DELTA_ABS_LOW = _get(_dc, "delta_abs_low", float)
DELTA_ABS_HIGH = _get(_dc, "delta_abs_high", float)
TTM_DAYS_LOW = _get(_dc, "ttm_days_low", int)
VEGA_MIN = _get(_dc, "vega_min", float)
# Applied when the tensor cache is read: zero out split-day return jumps.
SPLIT_LOG_RET_ABS_THRESHOLD = float(_dc.get("split_log_ret_abs_threshold", 0.20))

# Surface grid (paper grid: 11 non-uniform moneyness x 9 maturity = 99)
_sg = _sec("surface_grid")
SURFACE_M = int(_sg["surface_m"])                          # 11
SURFACE_T = int(_sg["surface_t"])                          # 9
SURFACE_DIM = SURFACE_M * SURFACE_T                        # 99
M_GRID = tuple(float(x) for x in _sg["m_grid"])
TAU_GRID_YEARS = tuple(float(x) for x in _sg["tau_grid_years"])
TAU_GRID_DAYS = tuple(round(t * 365.0, 2) for t in TAU_GRID_YEARS)   # report only

# Joint-generation vector dimensions
COND_EXTRA_DIM = int(_sec("model")["cond_extra_dim"])      # 3: R_{t-1}, R_{t-2}, RV
COND_DIM = COND_EXTRA_DIM + SURFACE_DIM                    # 102
TARGET_DIM = 1 + SURFACE_DIM                               # 100

# Chronological split
_sp = _sec("split")
TRAIN_START = str(_sp["train_start"])
TRAIN_END = str(_sp["train_end"])
TEST_START = str(_sp["test_start"])
TEST_END = str(_sp["test_end"])

# Nadaraya-Watson bandwidths: defaults + the per-ticker search that minimizes
# static arbitrage on the train split (ivs/data/prepare/bandwidth_search.py).
_bw = _sec("bandwidth")
KERNEL_H_M = _get(_bw, "kernel_h_m", float)
KERNEL_H_TAU = _get(_bw, "kernel_h_tau", float)
BANDWIDTH_SEARCH_LOW = _get(_bw, "search_low", float)
BANDWIDTH_SEARCH_HIGH = _get(_bw, "search_high", float)
BANDWIDTH_SEARCH_STEP = _get(_bw, "search_step", float)
BANDWIDTH_SEARCH_FIRST_N_DAYS = _get(_bw, "search_first_n_days", int)
BANDWIDTH_SEARCH_SEED = _get(_bw, "search_seed", int)

# Risk-free rate: put-call parity per trade date, with a T-bill fallback
_rt = _sec("rates")
RATE_SOURCE = _get(_rt, "source", str)                     # "parity" | "dgs3mo"
RATE_FALLBACK = _get(_rt, "fallback", str)
RATE_PARITY_MIN_PAIRS = _get(_rt, "parity_min_pairs", int)
RATE_PARITY_R_LOW = _get(_rt, "parity_r_low", float)
RATE_PARITY_R_HIGH = _get(_rt, "parity_r_high", float)

# Realized-vol / return features
RV_WINDOW = int(_sec("features")["rv_window"])             # 21
RV_ANNUALIZER_SQRT = (252.0 / RV_WINDOW) ** 0.5
RET_ANNUALIZER_SQRT = 252.0 ** 0.5

# Model architecture
#   diffusion : FiLM denoiser + DDPM schedule ("diffusion" section)
#   volgan    : tiny Softplus MLP generator/discriminator ("model" section)
_md = _sec("model")
_df = _sec("diffusion")
NUM_TIMESTEPS = _get(_df, "num_timesteps", int)
BETA_SCHEDULE = _get(_df, "beta_schedule", str)
NUM_BLOCKS = _get(_df, "num_blocks", int)
TIME_EMB_DIM = _get(_df, "time_emb_dim", int)
DROPOUT = _get(_df, "dropout", float)
EMA_DECAY = _get(_df, "ema_decay", float)
USE_EMA = _get(_df, "use_ema", bool)
NOISE_DIM = _get(_md, "noise_dim", int)                    # VolGAN only
# Hidden width lives in the family's own section (256 for the denoiser, 16 for
# the VolGAN generator whose second layer doubles it to 32).
HIDDEN_DIM = int(_df["hidden_dim"]) if "hidden_dim" in _df else _get(_md, "hidden_dim", int)

# Training
_tr = _sec("training")
EPOCHS = int(_tr["epochs"])
BATCH_SIZE = int(_tr["batch_size"])                        # PER-GPU batch under DDP
SEED = int(_tr.get("seed", GLOBAL_SEED))
# diffusion
LEARNING_RATE = _get(_tr, "learning_rate", float)
WEIGHT_DECAY = _get(_tr, "weight_decay", float)
GRADIENT_CLIP = _get(_tr, "gradient_clip", float)
WORLD_SIZE = int(_tr.get("world_size", 1))                 # torchrun --nproc_per_node
USE_AMP = bool(_tr.get("use_amp", False))
AMP_DTYPE = str(_tr.get("amp_dtype", "bfloat16"))
# volgan
LR_G = _get(_tr, "lr_g", float)
LR_D = _get(_tr, "lr_d", float)
N_GRAD_MATCH = _get(_tr, "n_grad_match", int)

_lg = _sec("logging")
LOG_EVERY_STEPS = int(_lg.get("log_every_steps", _tr.get("log_every_steps", 50)))
CKPT_EVERY_EPOCHS = int(_lg.get("ckpt_every_epochs", _tr.get("ckpt_every_epochs", 250)))

# Generation
_gn = _sec("generation")
SAMPLES_PER_STATE = int(_gn["samples_per_state"])          # S = 1000
BATCH_STATES = int(_gn["batch_states"])

# Scenario reweighting, w_i = exp(-beta*Phi_i) / sum_j exp(-beta*Phi_j)
#   fixed_beta  -> global temperature used for the return coverage (Table 3)
#   kl_constant -> C in the per-day beta = C * max_i w_i used for Table 2
_rw = _sec("reweight")
REWEIGHT_FIXED_BETA = float(_rw.get("fixed_beta", 50.0))
REWEIGHT_KL_CONSTANT = float(_rw.get("kl_constant", 500.0))

# Penalty-weight calibration (gradient matching, paper procedure)
_pc = _sec("penalty_calibration")
CALIB_REF_EPOCHS = int(_pc.get("ref_epochs", 25))
SMOOTH_TARGET_RATIO = float(_pc.get("smoothness_target_ratio", 1.0))
ARB_TARGET_RATIO = float(_pc.get("arbitrage_target_ratio", 1.0))
ARB_CLAMP_NORM = float(_pc.get("clamp_norm", 5.0))

# Version (loss-variant) configs
VERSION_CFG: dict = CONFIG["versions"]


def version_cfg(v: str) -> dict:
    return CONFIG["versions"][v.lower()]


DEFAULT_VERSION: str = VERSIONS[0]

# Parallelism
_hw = _sec("hardware")
N_JOBS = max(1, int((os.cpu_count() or 16) * float(_hw.get("n_jobs_fraction", 0.5))))
NUM_DATALOADER_WORKERS = int(_hw.get("num_dataloader_workers", 0))
TICKER_JOBS = int(_hw.get("ticker_jobs", 1))
PARQUET_COMPRESSION = str(_hw.get("parquet_compression", "zstd"))
PARQUET_COMPRESSION_LEVEL = int(_hw.get("parquet_compression_level", 3))


# Version-based artifact path helpers
def ckpt_dir(version: str) -> Path:
    p = CKPT_DIR / f"version_{version}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sample_dir(version: str, ticker: str) -> Path:
    p = SAMPLE_DIR / f"version_{version}" / ticker
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir(version: str) -> Path:
    p = LOG_DIR / f"version_{version}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def describe() -> str:
    """One-line banner printed by the entry points."""
    return (f"[config] experiment={EXPERIMENT} family={FAMILY} "
            f"version={DEFAULT_VERSION} tickers={len(TICKERS)} "
            f"grid={SURFACE_M}x{SURFACE_T} file={CONFIG_PATH}")
