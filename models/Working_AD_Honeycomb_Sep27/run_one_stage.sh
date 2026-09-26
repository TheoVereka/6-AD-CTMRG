#!/usr/bin/env bash
# One h=0, single-J2, pure-LBFGS continuation stage.
set -euo pipefail

: "${ALIAS:?ALIAS must be exported}"
: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${SEED_J2:?SEED_J2 must be exported}"
: "${TARGET_J2:?TARGET_J2 must be exported}"
: "${TEXTURE:?TEXTURE must be exported}"
: "${ORIENTATION:?ORIENTATION must be exported}"
: "${INSURANCE:?INSURANCE must be exported}"
: "${DIRECTION:?DIRECTION must be exported}"
: "${STAGE_INDEX:?STAGE_INDEX must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported}"
: "${PREVIOUS_CKPT:?PREVIOUS_CKPT must be exported}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported}"

[[ "${ALIAS}" =~ ^a0[1-4]$ ]] || { echo "invalid alias ${ALIAS}" >&2; exit 2; }
[[ "${INSURANCE}" =~ ^[12]$ ]] || { echo "invalid insurance ${INSURANCE}" >&2; exit 2; }
[[ "${TEXTURE}" == "plaquette" || "${TEXTURE}" == "dimer-plaquette" ]] || {
    echo "invalid texture ${TEXTURE}" >&2; exit 2;
}
[[ "${DIRECTION}" == "left" || "${DIRECTION}" == "right" ]] || {
    echo "invalid direction ${DIRECTION}" >&2; exit 2;
}
[[ "${ORIENTATION}" =~ ^[012]$ ]] || { echo "invalid orientation" >&2; exit 2; }

BUNDLE_DIR="${BUNDLE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
MAIN="${BUNDLE_DIR}/main_C3_LBFGS.py"
[[ -f "${MAIN}" && -f "${BUNDLE_DIR}/core_C3.py" ]] || {
    echo "self-contained numerical code is missing" >&2; exit 3;
}

mkdir -p "${OUTPUT_DIR}"
BEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
LATEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_latest.pt"
OBS_FILE="${OUTPUT_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]]; then
    echo "${ALIAS} ${DIRECTION} stage ${STAGE_INDEX}, J2=${TARGET_J2}: already complete"
    exit 0
fi

RESUME_CKPT="${PREVIOUS_CKPT}"
if [[ -s "${LATEST_CKPT}" ]]; then
    RESUME_CKPT="${LATEST_CKPT}"
elif [[ -s "${BEST_CKPT}" ]]; then
    RESUME_CKPT="${BEST_CKPT}"
fi
[[ -s "${RESUME_CKPT}" ]] || {
    echo "required predecessor tensor is missing: ${RESUME_CKPT}" >&2; exit 3;
}

ALIAS_NUMBER=$((10#${ALIAS#a}))
J2_DIGITS="${TARGET_J2#0.}"
DIRECTION_CODE=1
[[ "${DIRECTION}" == "right" ]] && DIRECTION_CODE=2
SEED_NUMBER=$((2700000 + 100000 * INSURANCE + 10000 * ALIAS_NUMBER + 1000 * DIRECTION_CODE + 10#${J2_DIGITS}))

echo "Alias=${ALIAS}; insurance=${INSURANCE}; ${DIRECTION} stage=${STAGE_INDEX}; seed J2=${SEED_J2}; target J2=${TARGET_J2}"
echo "Original isotropic Hamiltonian (h=0), pure L-BFGS, internal limit=${STAGE_HOURS} h"
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
    --gpu \
    --ngpu 1 \
    --fix-seed \
    --seed "${SEED_NUMBER}" \
    --output-dir "${OUTPUT_DIR}" \
    --no-mean-field-init \
    --resume "${RESUME_CKPT}" \
    --resume-tensors-only

if [[ ! -s "${BEST_CKPT}" || ! -s "${OBS_FILE}" ]]; then
    echo "stage did not finish cleanly in ${OUTPUT_DIR}" >&2
    exit 4
fi
echo "COMPLETED ${ALIAS} insurance ${INSURANCE} ${DIRECTION} stage ${STAGE_INDEX}, J2=${TARGET_J2}"
