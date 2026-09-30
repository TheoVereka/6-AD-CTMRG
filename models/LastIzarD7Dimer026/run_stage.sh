#!/usr/bin/env bash
# Execute one D=7, J2=.26 dimer-sector stage. Dependencies are submitted
# externally by submit_all.sh; this runner never submits jobs itself.
set -euo pipefail

: "${TASK_FAMILY:?TASK_FAMILY is required}"
: "${STAGE_KIND:?STAGE_KIND is required}"
: "${MAIN_KIND:?MAIN_KIND is required}"
: "${SEED_NUMBER:?SEED_NUMBER is required}"
: "${OUTPUT_REL:?OUTPUT_REL is required}"
: "${MEAN_FIELD_START:?MEAN_FIELD_START is required}"
: "${VBC_FIELD:?VBC_FIELD is required}"

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
cd "${BUNDLE_DIR}"

D=7
CHI=91
CTM_STEPS=50
J2=0.26
STAGE_HOURS=70

case "${MAIN_KIND}" in
    adam-lbfgs) MAIN="${BUNDLE_DIR}/main_C3.py" ;;
    lbfgs) MAIN="${BUNDLE_DIR}/main_C3_LBFGS.py" ;;
    *) echo "Invalid MAIN_KIND=${MAIN_KIND}" >&2; exit 2 ;;
esac
case "${STAGE_KIND}" in
    adiabatic|pin_h|pin_h0) ;;
    *) echo "Invalid STAGE_KIND=${STAGE_KIND}" >&2; exit 2 ;;
esac
[[ "${MEAN_FIELD_START}" == "0" || "${MEAN_FIELD_START}" == "1" ]] || {
    echo "MEAN_FIELD_START must be 0 or 1" >&2
    exit 2
}
[[ -s "${MAIN}" && -s "${BUNDLE_DIR}/core_C3.py" ]] || {
    echo "Missing bundled Python code" >&2
    exit 3
}

OUTPUT_DIR="${BUNDLE_DIR}/${OUTPUT_REL}"
mkdir -p "${OUTPUT_DIR}"
BEST_CKPT="${OUTPUT_DIR}/sweep_D7_chi91_best.pt"
LATEST_CKPT="${OUTPUT_DIR}/sweep_D7_chi91_latest.pt"
OBS_FILE="${OUTPUT_DIR}/D_7_chi_91_energy_magnetization_correlation.txt"
COMPLETION_MARKER="${OUTPUT_DIR}/COMPLETED.stage"

if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]]; then
    printf 'task=%s\ncompleted_utc=%s\n' \
        "${TASK_FAMILY}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${COMPLETION_MARKER}.tmp"
    mv -f "${COMPLETION_MARKER}.tmp" "${COMPLETION_MARKER}"
    echo "Already complete: ${TASK_FAMILY}"
    exit 0
fi

RESUME_CKPT=""
if [[ -s "${LATEST_CKPT}" ]]; then
    RESUME_CKPT="${LATEST_CKPT}"
    echo "Resuming interrupted stage from its latest checkpoint"
elif [[ -s "${BEST_CKPT}" ]]; then
    RESUME_CKPT="${BEST_CKPT}"
elif [[ -n "${INPUT_REL:-}" ]]; then
    RESUME_CKPT="${BUNDLE_DIR}/${INPUT_REL}"
fi
if [[ "${MEAN_FIELD_START}" == "0" && ! -s "${RESUME_CKPT}" ]]; then
    echo "Required predecessor/seed tensor is missing: ${RESUME_CKPT:-<unset>}" >&2
    exit 3
fi

echo "Task=${TASK_FAMILY}; stage=${STAGE_KIND}; J2=${J2}; D=${D}; chi=${CHI}"
echo "Main=${MAIN_KIND}; dimer pin h=${VBC_FIELD}; internal limit=${STAGE_HOURS} h"
echo "Output=${OUTPUT_DIR}"

ARGS=(
    --J2 "${J2}"
    --ansatz twoc3
    --Ds "${D}"
    --chi-min "${CHI}"
    --chi-max "${CHI}"
    --chi-step 1
    --ctm-max-steps "${CTM_STEPS}"
    --hours "${STAGE_HOURS}"
    --optimizer lbfgs
    --vbc-branch dimer-plaquette
    --vbc-orientation 0
    --vbc-field "${VBC_FIELD}"
    --noise 0.001
    --double
    --gpu
    --ngpu 1
    --fix-seed
    --seed "${SEED_NUMBER}"
    --output-dir "${OUTPUT_DIR}"
)
if [[ "${MAIN_KIND}" == "adam-lbfgs" ]]; then
    ARGS+=(--adam-warmup-lbfgs)
fi
if [[ -n "${RESUME_CKPT}" ]]; then
    ARGS+=(--no-mean-field-init --resume "${RESUME_CKPT}" --resume-tensors-only)
else
    ARGS+=(--mean-field-init)
fi

python -u "${MAIN}" "${ARGS[@]}"

if [[ ! -s "${BEST_CKPT}" || ! -s "${OBS_FILE}" ]]; then
    echo "Stage returned without a complete tensor+observation pair" >&2
    exit 4
fi
printf 'task=%s\ncompleted_utc=%s\n' \
    "${TASK_FAMILY}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${COMPLETION_MARKER}.tmp"
mv -f "${COMPLETION_MARKER}.tmp" "${COMPLETION_MARKER}"
echo "COMPLETED: ${TASK_FAMILY}: ${BEST_CKPT}"
