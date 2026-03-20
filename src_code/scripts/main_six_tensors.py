import argparse
import collections
import datetime
import json
import os
import sys
import time

# ── CPU threading ─────────────────────────────────────────────────────────────
# MUST be set BEFORE importing NumPy / PyTorch / MKL — they read these at init.
#
# Hardware: Intel Core i7-8665U — 4 physical cores × 2 HT = 8 logical CPUs.
# PyTorch is built with Intel MKL (BLAS/LAPACK) + OpenMP.
#
# Use 4 = physical core count.  Hyperthreading does NOT help SVD / dense
# matmul: both HTs on the same core share the same FPU and L1/L2 cache, so
# using 8 threads starves each other rather than doubling throughput.
#
# Use os.environ.setdefault so a user can still override from the shell:
#   OMP_NUM_THREADS=2 python scripts/main_sweep_CPU.py   ← respected
_N_PHYSICAL_CORES = 4
os.environ.setdefault("OMP_NUM_THREADS", str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_PHYSICAL_CORES))
# Prevent MKL from silently reducing thread count when it detects nested
# parallelism or high system load — we always want the full 4 threads.
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
# Pin each OpenMP thread to a distinct physical core (not a hyperthread sibling).
# granularity=fine  → thread-level pinning (finest possible)
# compact           → pack onto as few sockets as possible (single-socket here)
# 1                 → permute: fill one HT per core before doubling up (with
#                     4 threads on 4 cores this has no effect, but future-proofs)
# 0                 → offset: start from logical CPU 0
# Result: threads 0-3 map to cores 0-3, no migration between iterations.
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
# After a parallel region the OpenMP workers normally spin-wait for ~200 ms
# before going to sleep, burning a whole core.  Set to 0 so they yield
# immediately — the outer L-BFGS loop is sequential so this saves real time.
os.environ.setdefault("KMP_BLOCKTIME", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opt_einsum as oe
import torch


from core_six_tensors import (
    set_dtype,
)


USE_DOUBLE_PRECISION = True

USE_REAL_TENSORS = False




