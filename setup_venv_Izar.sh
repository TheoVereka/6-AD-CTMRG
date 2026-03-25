#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cluster venv setup for 6-AD-CTMRG
# Target: EPFL Izar (izar.hpc.epfl.ch) — GPU cluster, V100 (CC 7.0), CUDA 11.8
#
# NOTE: Python 3.11 is not available on Izar. Python 3.10.4 is used instead.
#       The codebase is compatible with Python 3.10+ (PEP 604 union types).
#
# If the module loads below fail with "cannot be loaded as requested", run:
#   module spider python/3.10.4
# and prepend whatever prerequisite module it lists (e.g. gcc/X.Y.Z) before
# the python line, then re-run this script.
#
# Usage:
#   bash setup_venv_Izar.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR="${HOME}/venvs/6adctmrg_Izar"

# ── 1. Load modules ───────────────────────────────────────────────────────────
# cuda/11.8.0: matches torch==2.5.1+cu118; V100 CC 7.0 is fully supported.
# python/3.10.4: highest available Python on Izar.
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.10.4

if ! python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
    echo "ERROR: Python 3.10+ not active after module load — check prerequisites."
    echo "Run:   module spider python/3.10.4"
    exit 1
fi
echo "Using: $(python3 --version)"
echo "CUDA:  $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH (OK — PyTorch bundles its own runtime)')"

# ── 2. Create venv ────────────────────────────────────────────────────────────
[[ -d "${VENV_DIR}" ]] && { echo "Removing old venv ..."; rm -rf "${VENV_DIR}"; }
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel

# ── 3. PyTorch 2.5.1 + CUDA 11.8 ─────────────────────────────────────────────
# cuDNN is bundled inside the wheel — no separate cuDNN module needed.
# V100 (CC 7.0) is fully supported by this wheel.
pip install torch==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

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
# Run on a GPU node for the CUDA check:
#   srun --partition=gpu --gres=gpu:1 --pty bash -c \
#     "source ~/venvs/6adctmrg/bin/activate && python -c 'import torch; print(torch.cuda.is_available())'"
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
print(f"  → False here is normal; GPU nodes will show True")
x = torch.randn(4, 4, dtype=torch.complex64)
print(f"Complex64 matmul OK: {torch.linalg.norm(x @ x).item():.4f}")
EOF

echo ""
echo "Done.  Activate with:  source ${VENV_DIR}/bin/activate"
echo ""
echo "To verify GPU on a compute node:"
echo "  srun --partition=gpu --gres=gpu:1 --pty bash -c \\"
echo "    'module load gcc/11.3.0 &&module load cuda/11.8.0 python/3.10.4 && source ~/venvs/6adctmrg_Izar/bin/activate && python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
