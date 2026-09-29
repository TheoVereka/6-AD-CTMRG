#!/usr/bin/env bash
# Run exactly one physical optimization stage. Slurm dependencies are created
# by submit_all.sh; this script never submits another job.
set -euo pipefail

: "${TASK_FAMILY:?TASK_FAMILY must be exported by the static .run file}"
: "${STAGE_KIND:?STAGE_KIND must be exported by the static .run file}"
: "${MAIN_KIND:?MAIN_KIND must be adam-lbfgs or lbfgs}"
: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${CTM_STEPS:?CTM_STEPS must be exported}"
: "${J2:?J2 must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported}"
: "${VBC_FIELD:?VBC_FIELD must be exported}"
: "${MEAN_FIELD_START:?MEAN_FIELD_START must be 0 or 1}"
: "${SEED_NUMBER:?SEED_NUMBER must be exported}"
: "${OUTPUT_REL:?OUTPUT_REL must be exported}"

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
cd "${BUNDLE_DIR}"

case "${D}:${CHI}:${CTM_STEPS}" in
    5:50:70|6:72:130) ;;
    *)
        echo "Refusing unexpected numerical configuration D=${D}, chi=${CHI}, CTM_STEPS=${CTM_STEPS}" >&2
        exit 70
        ;;
esac
case "${MAIN_KIND}" in
    adam-lbfgs) MAIN="${BUNDLE_DIR}/main_C3.py" ;;
    lbfgs)      MAIN="${BUNDLE_DIR}/main_C3_LBFGS.py" ;;
    *) echo "Invalid MAIN_KIND=${MAIN_KIND}" >&2; exit 2 ;;
esac
case "${STAGE_KIND}" in
    adiabatic|pin_h005|pin_h0) ;;
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
BEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
LATEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_latest.pt"
OBS_FILE="${OUTPUT_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"

if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]]; then
    echo "Already complete: ${TASK_FAMILY}, J2=${J2}, ${OUTPUT_DIR}"
    exit 0
fi

# On manual resubmission, resume the interrupted current stage. Otherwise use
# the immutable seed or the best tensor produced by the afterok predecessor.
RESUME_CKPT=""
if [[ -s "${LATEST_CKPT}" ]]; then
    RESUME_CKPT="${LATEST_CKPT}"
    echo "Resuming interrupted current stage from latest checkpoint"
elif [[ -s "${BEST_CKPT}" ]]; then
    RESUME_CKPT="${BEST_CKPT}"
    echo "Resuming current stage from its existing best checkpoint"
elif [[ -n "${INPUT_REL:-}" ]]; then
    RESUME_CKPT="${BUNDLE_DIR}/${INPUT_REL}"
fi

if [[ "${MEAN_FIELD_START}" == "0" && ! -s "${RESUME_CKPT}" ]]; then
    echo "Required predecessor/seed tensor is missing: ${RESUME_CKPT:-<unset>}" >&2
    exit 3
fi
if [[ -n "${RESUME_CKPT}" && ! -s "${RESUME_CKPT}" ]]; then
    echo "Configured input tensor is missing: ${RESUME_CKPT}" >&2
    exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
fi
echo "Task=${TASK_FAMILY}; stage=${STAGE_KIND}; J2=${J2}; D=${D}; chi=${CHI}"
echo "Main=${MAIN_KIND}; h=${VBC_FIELD}; internal limit=${STAGE_HOURS} h"
echo "Output=${OUTPUT_DIR}"

COMMON_ARGS=(
    --J2 "${J2}"
    --ansatz twoc3
    --Ds "${D}"
    --chi-min "${CHI}"
    --chi-max "${CHI}"
    --chi-step 1
    --ctm-max-steps "${CTM_STEPS}"
    --hours "${STAGE_HOURS}"
    --optimizer lbfgs
    --vbc-branch plaquette
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
    COMMON_ARGS+=(--adam-warmup-lbfgs)
fi
if [[ -n "${RESUME_CKPT}" ]]; then
    COMMON_ARGS+=(--no-mean-field-init --resume "${RESUME_CKPT}" --resume-tensors-only)
else
    COMMON_ARGS+=(--mean-field-init)
fi

python -u "${MAIN}" "${COMMON_ARGS[@]}"

if [[ ! -s "${BEST_CKPT}" || ! -s "${OBS_FILE}" ]]; then
    echo "Stage returned without a complete tensor+observables pair: ${OUTPUT_DIR}" >&2
    exit 4
fi
echo "COMPLETED: ${TASK_FAMILY}, J2=${J2}, ${BEST_CKPT}"
