#!/usr/bin/env bash
# Optimize exactly one J2 point. Slurm dependencies connect these stages.
set -euo pipefail

: "${SEED_ID:?SEED_ID must be exported}"
: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${SEED_J2:?SEED_J2 must be exported}"
: "${TARGET_J2:?TARGET_J2 must be exported}"
: "${TEXTURE:?TEXTURE must be exported}"
: "${ORIENTATION:?ORIENTATION must be exported}"
: "${DIRECTION:?DIRECTION must be exported}"
: "${INSURANCE:?INSURANCE must be exported}"
: "${STAGE_INDEX:?STAGE_INDEX must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported}"
: "${PREVIOUS_CKPT:?PREVIOUS_CKPT must be exported}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported}"

[[ "${TEXTURE}" == "plaquette" || "${TEXTURE}" == "dimer-plaquette" ]] || {
    echo "Invalid texture: ${TEXTURE}" >&2
    exit 2
}
[[ "${DIRECTION}" == "left" || "${DIRECTION}" == "right" ]] || {
    echo "Invalid direction: ${DIRECTION}" >&2
    exit 2
}
[[ "${ORIENTATION}" =~ ^[012]$ ]] || {
    echo "Invalid orientation: ${ORIENTATION}" >&2
    exit 2
}
[[ "${INSURANCE}" =~ ^[12]$ ]] || {
    echo "Insurance copy must be 1 or 2" >&2
    exit 2
}

BUNDLE_DIR="${BUNDLE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
CODE_DIR="${CODE_DIR:-/home/chye/VBCPinningLyraLBFGS}"
CODE_DIR="$(cd -- "${CODE_DIR}" && pwd)"
MAIN="${CODE_DIR}/main_C3_LBFGS.py"
MEMORY_GUARD="${CODE_DIR}/izar_memory_guard.sh"
[[ -f "${MAIN}" ]] || { echo "Missing ${MAIN}" >&2; exit 3; }
[[ -f "${CODE_DIR}/core_C3.py" ]] || { echo "Missing core_C3.py in ${CODE_DIR}" >&2; exit 3; }
[[ -f "${MEMORY_GUARD}" ]] || { echo "Missing ${MEMORY_GUARD}" >&2; exit 3; }

export D CHI
bash "${MEMORY_GUARD}"
mkdir -p "${OUTPUT_DIR}"
BEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
LATEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_latest.pt"
OBS_FILE="${OUTPUT_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"

if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]]; then
    echo "J2=${TARGET_J2} already complete: ${BEST_CKPT}"
    exit 0
fi

# A prior interrupted submission resumes its own current-stage checkpoint.
# Otherwise, afterok guarantees PREVIOUS_CKPT was produced successfully by
# the preceding J2 job (or is the immutable manually selected seed).
RESUME_CKPT="${PREVIOUS_CKPT}"
if [[ -s "${LATEST_CKPT}" ]]; then
    echo "Resuming interrupted J2=${TARGET_J2} from its latest checkpoint"
    RESUME_CKPT="${LATEST_CKPT}"
elif [[ -s "${BEST_CKPT}" ]]; then
    echo "Resuming J2=${TARGET_J2} from its existing best checkpoint"
    RESUME_CKPT="${BEST_CKPT}"
fi
[[ -s "${RESUME_CKPT}" ]] || {
    echo "Required predecessor tensor is missing: ${RESUME_CKPT}" >&2
    exit 3
}

J2_DIGITS="${TARGET_J2#0.}"
DIRECTION_CODE=1
[[ "${DIRECTION}" == "right" ]] && DIRECTION_CODE=2
SEED_NUMBER=$((2000000 + 100000 * INSURANCE + 10000 * D \
    + 100 * (10#${SEED_ID#s}) + 10 * DIRECTION_CODE + 10#${J2_DIGITS}))

echo "Single Slurm stage: ${SEED_ID} ${DIRECTION} insurance=${INSURANCE} index=${STAGE_INDEX}"
echo "Original Hamiltonian: seed J2=${SEED_J2} -> target J2=${TARGET_J2}, h=0"
echo "D=${D}, chi=${CHI}, pure L-BFGS, internal limit=${STAGE_HOURS} h"
echo "Resume tensor: ${RESUME_CKPT}"

python "${MAIN}" \
    --J2 "${TARGET_J2}" \
    --ansatz twoc3 \
    --Ds "${D}" \
    --chi-min "${CHI}" \
    --chi-max "${CHI}" \
    --chi-step 1 \
    --hours "${STAGE_HOURS}" \
    --optimizer lbfgs \
    --vbc-branch "${TEXTURE}" \
    --vbc-orientation "${ORIENTATION}" \
    --vbc-field 0 \
    --noise 0.001 \
    --fix-seed \
    --seed "${SEED_NUMBER}" \
    --output-dir "${OUTPUT_DIR}" \
    --no-mean-field-init \
    --resume "${RESUME_CKPT}" \
    --resume-tensors-only

if [[ ! -s "${BEST_CKPT}" || ! -s "${OBS_FILE}" ]]; then
    echo "J2=${TARGET_J2} did not finish cleanly in ${OUTPUT_DIR}" >&2
    exit 4
fi
echo "COMPLETED J2=${TARGET_J2}: ${BEST_CKPT}"

