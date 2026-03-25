#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cluster venv setup for 6-AD-CTMRG
# Target: EPFL Kuma (kuma.hpc.epfl.ch) — GPU cluster
#         H100 (CC 9.0, 94 GB) on partition h100
#         L40s (CC 8.9, 48 GB) on partition l40s
#
# Module load order matters: cuda/12.4.1 is a prerequisite for python/3.11.7.
# PyTorch wheel: cu124 (CUDA 12.4) — no cu125 wheel exists from PyTorch.
#
# Usage:
#   bash setup_venv_Kuma.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR="${HOME}/venvs/6adctmrg_Kuma"

# ── 1. Load modules ───────────────────────────────────────────────────────────
# cuda/12.4.1 must be loaded FIRST — it is a prerequisite for python/3.11.7.
module load gcc/13.2.0
module load cuda/12.4.1
module load python/3.11.7

if ! python3 -c "import sys; assert sys.version_info[:2]==(3,11)" 2>/dev/null; then
    echo "ERROR: Python 3.11 not active after module load — check prerequisites."
    echo "Run:   module spider python/3.11.7"
    exit 1
fi
echo "Using: $(python3 --version)"
echo "CUDA:  $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH (OK)')"

# ── 2. Create venv ────────────────────────────────────────────────────────────
[[ -d "${VENV_DIR}" ]] && { echo "Removing old venv ..."; rm -rf "${VENV_DIR}"; }
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel

# ── 3. PyTorch 2.5.1 + CUDA 12.4 ─────────────────────────────────────────────
# cu124 is the closest available wheel to CUDA 12.4/12.5.
# cuDNN is bundled — no separate cuDNN module needed.
# H100 (CC 9.0) and L40s (CC 8.9) are both fully supported.
pip install torch==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 4. Scientific stack ───────────────────────────────────────────────────────
pip install \
    numpy==2.1.3      \
    scipy==1.14.1     \
    opt_einsum==3.4.0 \
    matplotlib==3.9.4 \
    tqdm==4.67.1      \
    h5py==3.12.1      \
    pytest==8.4.2

# ── 5. Smoke-test (login node — CUDA not available here) ──────────────────────
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
print(f"CUDA available (login node): {torch.cuda.is_available()}")
print(f"  -> False here is normal; GPU nodes will show True")
x = torch.randn(4, 4, dtype=torch.complex64)
print(f"Complex64 matmul OK: {torch.linalg.norm(x @ x).item():.4f}")
EOF

echo ""
echo "Done.  Activate with:  source ${VENV_DIR}/bin/activate"
echo ""
echo "To verify GPU on a compute node (H100):"
echo "  srun --partition=h100 --gres=gpu:1 --pty bash -c \\"
echo "    'module load gcc/13.2.0 && module load cuda/12.4.1 python/3.11.7 && source ~/venvs/6adctmrg_Kuma/bin/activate && python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
echo ""
echo "To verify GPU on a compute node (L40s):"
echo "  srun --partition=l40s --gres=gpu:1 --pty bash -c \\"
echo "    'module load gcc/13.2.0 && module load cuda/12.4.1 python/3.11.7 && source ~/venvs/6adctmrg_Kuma/bin/activate && python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
