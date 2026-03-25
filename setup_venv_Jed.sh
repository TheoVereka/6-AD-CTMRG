#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cluster venv setup for 6-AD-CTMRG
# Target: EPFL Jed (jed.hpc.epfl.ch) — CPU-only cluster, no GPU.
#
# Usage:
#   module load gcc/13.2.0 python/3.11.7
#   bash setup_venv_Jed.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR="${HOME}/venvs/6adctmrg"

# ── 1. Load prerequisite modules and require Python 3.11 ────────────────────
# gcc/13.2.0 must be loaded before python/3.11.7 on Jed.
module load gcc/13.2.0
module load python/3.11.7

if ! python3 -c "import sys; assert sys.version_info[:2]==(3,11)" 2>/dev/null; then
    echo "ERROR: Python 3.11 not active after module load — check module names."
    exit 1
fi
echo "Using: $(python3 --version)"

# ── 2. Create venv ────────────────────────────────────────────────────────────
[[ -d "${VENV_DIR}" ]] && { echo "Removing old venv ..."; rm -rf "${VENV_DIR}"; }
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel

# ── 3. PyTorch 2.5.1 — CPU only (Jed has no GPU) ────────────────────────────
pip install torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── 4. Scientific stack ───────────────────────────────────────────────────────
pip install \
    numpy==2.1.3      \
    scipy==1.14.1     \
    opt_einsum==3.4.0 \
    matplotlib==3.9.4 \
    tqdm==4.67.1      \
    h5py==3.12.1      \
    pytest==8.4.2

# ── 5. Smoke-test ─────────────────────────────────────────────────────────────
python - <<'EOF'
import torch, numpy, scipy, opt_einsum, matplotlib, tqdm, h5py, pytest
print(f"torch      {torch.__version__}")
print(f"numpy      {numpy.__version__}")
print(f"scipy      {scipy.__version__}")
print(f"opt_einsum {opt_einsum.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"tqdm       {tqdm.__version__}")
print(f"h5py       {h5py.__version__}")
print(f"pytest     {pytest.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}  (expected: False on Jed)")
x = torch.randn(4, 4, dtype=torch.complex64)
print(f"Complex64 matmul OK: {torch.linalg.norm(x @ x).item():.4f}")
EOF

echo ""
echo "Done.  Activate with:  source ${VENV_DIR}/bin/activate"
