#!/usr/bin/env bash

# Common implementation; jobs/*.run supplies one J2 and one environment pair.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash run_pair.sh J2 PAIR" >&2
    exit 2
fi
J2="$1"
PAIR="$2"
case "${J2}" in
    0.26|0.25|0.265|0.32|0.20|0.27|0.245|0.275|0.24|0.28) ;;
    *) echo "No bundled D8 seed for J2=${J2}" >&2; exit 2 ;;
esac
case "${PAIR}" in
    1|2|3) ;;
    *) echo "PAIR must be 1, 2, or 3" >&2; exit 2 ;;
esac

if [[ -n "${BUNDLE_DIR:-}" ]]; then
    BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
elif [[ -f "${PWD}/run_pair.sh" ]]; then
    BUNDLE_DIR="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_pair.sh" ]]; then
    BUNDLE_DIR="${SLURM_SUBMIT_DIR}"
else
    BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
export BUNDLE_DIR
cd "${BUNDLE_DIR}"
TAG="${J2/./p}"
CHECKPOINT="${BUNDLE_DIR}/seeds/J2_${TAG}/D_8/tensor_best.pt"
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results}"
RUN_DIR="${OUTROOT}/J2_${TAG}/D_8/chi_80/pair_${PAIR}"
mkdir -p "${RUN_DIR}"

module load gcc/11.3.0
module load cuda/11.8.0
source /home/chye/venvs/6adctmrg_Izar/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "START $(date -Is): J2=${J2} D=8 chi=80 pair=${PAIR} basis=64 batch=2 float64"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

python -u "${BUNDLE_DIR}/code/renyi2_twoc3.py" \
    --checkpoint "${CHECKPOINT}" \
    --J1 1 --J2 "${J2}" --pair "${PAIR}" \
    --chi 80 --device cuda \
    --subspace 64 --block-size 4 --batch 2 \
    --modes 8 16 32 --eig-tol 1e-9 --entropy-tol 1e-4 \
    --L-min 100 --L-max 1000 --L-step 2 \
    --ctm-max-steps 200 --ctm-retries 2 \
    --ctm-conv-mode both --ctm-conv-tol 1e-7 \
    --ctm-e-conv-threshold 2e-8 --rsvd-mode full_svd \
    --max-matvec 2000 --progress-every 10 --threads 1 \
    --seed 20261009 \
    --save-edges "${RUN_DIR}/edges.npz" \
    --output "${RUN_DIR}/result.json"

echo "FINISHED $(date -Is): ${RUN_DIR}"
