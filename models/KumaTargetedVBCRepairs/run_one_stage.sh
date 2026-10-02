#!/usr/bin/env bash
# One pure-LBFGS stage of a targeted h=0.02 -> h=0 Kuma repair chain.
set -euo pipefail

: "${ALIAS:?ALIAS must be exported}"
: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${SEED_J2:?SEED_J2 must be exported}"
: "${TARGET_J2:?TARGET_J2 must be exported}"
: "${TEXTURE:?TEXTURE must be exported}"
: "${ORIENTATION:?ORIENTATION must be exported}"
: "${FIELD:?FIELD must be exported}"
: "${FIELD_LABEL:?FIELD_LABEL must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported}"
: "${RANDOM_SEED:?RANDOM_SEED must be exported}"
: "${INPUT_CKPT:?INPUT_CKPT must be exported}"

[[ "${ALIAS}" =~ ^r0[1-3]$ ]] || { echo "invalid alias ${ALIAS}" >&2; exit 2; }
[[ "${TEXTURE}" == "plaquette" || "${TEXTURE}" == "dimer-plaquette" ]] || {
    echo "invalid texture ${TEXTURE}" >&2; exit 2;
}
[[ "${ORIENTATION}" =~ ^[012]$ ]] || { echo "invalid orientation" >&2; exit 2; }
[[ "${FIELD}" == "0.02" || "${FIELD}" == "0" ]] || {
    echo "invalid field ${FIELD}" >&2; exit 2;
}

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${BUNDLE_DIR}/main_C3_LBFGS.py"
OUTDIR="${BUNDLE_DIR}/Results_Kuma_TargetedRepairs/${ALIAS}/h_${FIELD_LABEL}"
BEST_CKPT="${OUTDIR}/sweep_D${D}_chi${CHI}_best.pt"
LATEST_CKPT="${OUTDIR}/sweep_D${D}_chi${CHI}_latest.pt"
OBS_FILE="${OUTDIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
COMPLETE="${OUTDIR}/COMPLETED.stage"

[[ -f "${MAIN}" && -f "${BUNDLE_DIR}/core_C3.py" ]] || {
    echo "self-contained numerical code is missing" >&2; exit 3;
}
mkdir -p "${OUTDIR}"

if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]]; then
    if [[ ! -s "${COMPLETE}" ]]; then
        printf '%s\n' "completed ${ALIAS} J2=${TARGET_J2} D=${D} h=${FIELD}" \
            > "${COMPLETE}.tmp"
        mv -f -- "${COMPLETE}.tmp" "${COMPLETE}"
    fi
    echo "Stage already complete: ${OUTDIR}"
    exit 0
fi

RESUME_CKPT="${INPUT_CKPT}"
if [[ -s "${LATEST_CKPT}" ]]; then
    RESUME_CKPT="${LATEST_CKPT}"
elif [[ -s "${BEST_CKPT}" ]]; then
    RESUME_CKPT="${BEST_CKPT}"
fi
[[ -s "${RESUME_CKPT}" ]] || {
    echo "required input tensor is missing: ${RESUME_CKPT}" >&2; exit 4;
}

echo "alias=${ALIAS} D=${D} chi=${CHI} seed_J2=${SEED_J2} target_J2=${TARGET_J2}"
echo "texture=${TEXTURE} orientation=${ORIENTATION} h=${FIELD} input=${RESUME_CKPT}"
echo "pure L-BFGS; internal limit=${STAGE_HOURS} h; CTM/tensor noise <= 1e-3"

python "${MAIN}" \
    --J2 "${TARGET_J2}" \
    --ansatz twoc3 \
    --Ds "${D}" \
    --chi-min "${CHI}" \
    --chi-max "${CHI}" \
    --chi-step 1 \
    --hours "${STAGE_HOURS}" \
    --ctm-max-steps 70 \
    --optimizer lbfgs \
    --vbc-branch "${TEXTURE}" \
    --vbc-orientation "${ORIENTATION}" \
    --vbc-field "${FIELD}" \
    --noise 0.001 \
    --gpu \
    --ngpu 1 \
    --fix-seed \
    --seed "${RANDOM_SEED}" \
    --output-dir "${OUTDIR}" \
    --no-mean-field-init \
    --resume "${RESUME_CKPT}" \
    --resume-tensors-only

[[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" ]] || {
    echo "stage did not finish cleanly: ${OUTDIR}" >&2; exit 5;
}
printf '%s\n' "completed ${ALIAS} J2=${TARGET_J2} D=${D} h=${FIELD}" \
    > "${COMPLETE}.tmp"
mv -f -- "${COMPLETE}.tmp" "${COMPLETE}"
echo "COMPLETED ${ALIAS} h=${FIELD}"
