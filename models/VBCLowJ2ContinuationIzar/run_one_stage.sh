#!/usr/bin/env bash
# Run one fixed-(D,h,J2) pure-LBFGS stage.  Slurm dependencies are submitted
# by submit_all.sh; this file never submits another job.
set -euo pipefail

: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${ORIENTATION:?ORIENTATION must be exported}"
: "${BRANCH:?BRANCH must be exported}"
: "${SIGNED_H:?SIGNED_H must be exported}"
: "${FIELD:?FIELD must be exported as a positive magnitude}"
: "${TARGET_J2:?TARGET_J2 must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported}"
: "${INPUT_REL:?INPUT_REL must be exported}"
: "${OUTPUT_REL:?OUTPUT_REL must be exported}"
: "${RANDOM_SEED:?RANDOM_SEED must be exported}"

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
cd "${BUNDLE_DIR}"

case "${D}:${CHI}" in
    6:108|7:126) ;;
    *)
        echo "Refusing D=${D}, chi=${CHI}; Izar D=8 is permanently banned, expected 6:108 or 7:126" >&2
        exit 70
        ;;
esac
case "${BRANCH}:${SIGNED_H}" in
    dimer-plaquette:-*) ;;
    plaquette:+*|plaquette:0.*) ;;
    *) echo "BRANCH=${BRANCH} is inconsistent with SIGNED_H=${SIGNED_H}" >&2; exit 2 ;;
esac
[[ "${ORIENTATION}" =~ ^[012]$ ]] || { echo "Bad orientation" >&2; exit 2; }
[[ -s "${BUNDLE_DIR}/main_C3_LBFGS.py" && -s "${BUNDLE_DIR}/core_C3.py" ]] || {
    echo "Bundled numerical code is missing" >&2; exit 3;
}

OUTPUT_DIR="${BUNDLE_DIR}/${OUTPUT_REL}"
INPUT_CKPT="${BUNDLE_DIR}/${INPUT_REL}"
mkdir -p "${OUTPUT_DIR}"
BEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
LATEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_latest.pt"
OBS_FILE="${OUTPUT_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"

if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]]; then
    echo "Already complete: D=${D}, signed h=${SIGNED_H}, J2=${TARGET_J2}"
    exit 0
fi
if [[ "${FORCE_INPUT_CHECKPOINT:-0}" == "1" ]]; then
    # Recovery jobs must restart from the explicitly selected last completed
    # J2 tensor, never from a partial checkpoint left by a failed target job.
    RESUME_CKPT="${INPUT_CKPT}"
elif [[ -s "${LATEST_CKPT}" ]]; then
    RESUME_CKPT="${LATEST_CKPT}"
elif [[ -s "${BEST_CKPT}" ]]; then
    RESUME_CKPT="${BEST_CKPT}"
else
    RESUME_CKPT="${INPUT_CKPT}"
fi
[[ -s "${RESUME_CKPT}" ]] || {
    echo "Required seed/predecessor tensor is missing: ${RESUME_CKPT}" >&2
    exit 3
}

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
fi
echo "D=${D}; chi=${CHI}; J2=${TARGET_J2}; signed h=${SIGNED_H}"
echo "branch=${BRANCH}; field magnitude=${FIELD}; orientation=${ORIENTATION}"
echo "resume=${RESUME_CKPT}; output=${OUTPUT_DIR}; pure L-BFGS"
[[ "${FORCE_INPUT_CHECKPOINT:-0}" == "1" ]] && \
    echo "recovery mode: ignored any target-stage partial checkpoint"

python -u "${BUNDLE_DIR}/main_C3_LBFGS.py" \
    --J2 "${TARGET_J2}" \
    --ansatz twoc3 \
    --Ds "${D}" \
    --chi-min "${CHI}" \
    --chi-max "${CHI}" \
    --chi-step 1 \
    --ctm-max-steps 70 \
    --hours "${STAGE_HOURS}" \
    --optimizer lbfgs \
    --vbc-branch "${BRANCH}" \
    --vbc-orientation "${ORIENTATION}" \
    --vbc-field "${FIELD}" \
    --noise 0.001 \
    --double \
    --gpu \
    --ngpu 1 \
    --fix-seed \
    --seed "${RANDOM_SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --no-mean-field-init \
    --resume "${RESUME_CKPT}" \
    --resume-tensors-only

[[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]] || {
    echo "Stage returned without a complete tensor+observables pair" >&2
    exit 4
}
echo "COMPLETED D=${D}, signed h=${SIGNED_H}, J2=${TARGET_J2}"
