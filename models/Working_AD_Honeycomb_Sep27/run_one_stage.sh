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
: "${DIRECTION:?DIRECTION must be exported}"
: "${STAGE_INDEX:?STAGE_INDEX must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported}"
: "${PREVIOUS_CKPT:?PREVIOUS_CKPT must be exported}"
: "${IMMUTABLE_SEED_CKPT:?IMMUTABLE_SEED_CKPT must be exported}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported}"

[[ "${ALIAS}" =~ ^a0[1-3]$ ]] || { echo "invalid alias ${ALIAS}" >&2; exit 2; }
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
ETA_GUARD="${BUNDLE_DIR}/eta_guard.py"
[[ -f "${MAIN}" && -f "${BUNDLE_DIR}/core_C3.py" && -f "${ETA_GUARD}" ]] || {
    echo "self-contained numerical code is missing" >&2; exit 3;
}

mkdir -p "${OUTPUT_DIR}"
BEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
LATEST_CKPT="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_latest.pt"
OBS_FILE="${OUTPUT_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
NEXT_CKPT="${OUTPUT_DIR}/resume_for_next.pt"
DECISION_FILE="${OUTPUT_DIR}/resume_decision.txt"
ETA_THRESHOLD="${ETA_THRESHOLD:-0.35}"
if [[ -s "${BEST_CKPT}" && -s "${OBS_FILE}" && -s "${NEXT_CKPT}" && -s "${DECISION_FILE}" ]]; then
    echo "${ALIAS} ${DIRECTION} stage ${STAGE_INDEX}, J2=${TARGET_J2}: already complete"
    exit 0
fi

ALIAS_NUMBER=$((10#${ALIAS#a}))
J2_DIGITS="${TARGET_J2#0.}"
DIRECTION_CODE=1
[[ "${DIRECTION}" == "right" ]] && DIRECTION_CODE=2
SEED_NUMBER=$((2700000 + 10000 * ALIAS_NUMBER + 1000 * DIRECTION_CODE + 10#${J2_DIGITS}))

run_attempt() {
    local attempt_dir="$1"
    local input_ckpt="$2"
    local random_seed="$3"
    local attempt_best="${attempt_dir}/sweep_D${D}_chi${CHI}_best.pt"
    local attempt_latest="${attempt_dir}/sweep_D${D}_chi${CHI}_latest.pt"
    local attempt_obs="${attempt_dir}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
    mkdir -p "${attempt_dir}"
    if [[ -s "${attempt_best}" && -s "${attempt_obs}" ]]; then
        echo "Completed attempt already exists: ${attempt_dir}"
        return 0
    fi
    local resume_ckpt="${input_ckpt}"
    if [[ -s "${attempt_latest}" ]]; then
        resume_ckpt="${attempt_latest}"
    elif [[ -s "${attempt_best}" ]]; then
        resume_ckpt="${attempt_best}"
    fi
    [[ -s "${resume_ckpt}" ]] || {
        echo "required input tensor is missing: ${resume_ckpt}" >&2; return 3;
    }
    echo "Attempt output=${attempt_dir}; resume=${resume_ckpt}; RNG=${random_seed}"
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
        --seed "${random_seed}" \
        --output-dir "${attempt_dir}" \
        --no-mean-field-init \
        --resume "${resume_ckpt}" \
        --resume-tensors-only
    [[ -s "${attempt_best}" && -s "${attempt_obs}" ]] || {
        echo "attempt did not finish cleanly in ${attempt_dir}" >&2; return 4;
    }
}

eta_is_accepted() {
    local observation="$1"
    local report="$2"
    local status
    if python "${ETA_GUARD}" "${observation}" --threshold "${ETA_THRESHOLD}" --report "${report}"; then
        return 0
    else
        status=$?
        [[ "${status}" == "10" ]] && return 10
        return "${status}"
    fi
}

echo "Alias=${ALIAS}; ${DIRECTION} stage=${STAGE_INDEX}; seed J2=${SEED_J2}; target J2=${TARGET_J2}"
echo "Original isotropic Hamiltonian (h=0), pure L-BFGS, internal limit=${STAGE_HOURS} h"
echo "Continuation guard: require |eta| >= ${ETA_THRESHOLD}"

INITIAL_INPUT="${PREVIOUS_CKPT}"
if [[ ! -s "${INITIAL_INPUT}" ]]; then
    echo "Previous-stage tensor is absent; using immutable branch seed for this attempt"
    INITIAL_INPUT="${IMMUTABLE_SEED_CKPT}"
fi
[[ -s "${INITIAL_INPUT}" ]] || {
    echo "immutable branch seed is also missing: ${IMMUTABLE_SEED_CKPT}" >&2; exit 3;
}
run_attempt "${OUTPUT_DIR}" "${INITIAL_INPUT}" "${SEED_NUMBER}"
if eta_is_accepted "${OBS_FILE}" "${OUTPUT_DIR}/eta_guard.json"; then
    cp -f -- "${BEST_CKPT}" "${NEXT_CKPT}"
    printf '%s\n' "accepted current J2=${TARGET_J2}: |eta| >= ${ETA_THRESHOLD}" > "${DECISION_FILE}"
else
    ETA_STATUS=$?
    [[ "${ETA_STATUS}" == "10" ]] || exit "${ETA_STATUS}"
    if (( STAGE_INDEX > 1 )) && [[ -s "${PREVIOUS_CKPT}" ]]; then
        cp -f -- "${PREVIOUS_CKPT}" "${NEXT_CKPT}"
        printf '%s\n' "rejected current J2=${TARGET_J2}: |eta| < ${ETA_THRESHOLD}; next J2 uses previous-stage tensor" > "${DECISION_FILE}"
    else
        RETRY_DIR="${OUTPUT_DIR}/retry_1"
        RETRY_BEST="${RETRY_DIR}/sweep_D${D}_chi${CHI}_best.pt"
        RETRY_OBS="${RETRY_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
        echo "No usable earlier J2-stage tensor; recomputing J2=${TARGET_J2} once"
        run_attempt "${RETRY_DIR}" "${INITIAL_INPUT}" "$((SEED_NUMBER + 500000))"
        if eta_is_accepted "${RETRY_OBS}" "${RETRY_DIR}/eta_guard.json"; then
            cp -f -- "${RETRY_BEST}" "${NEXT_CKPT}"
            printf '%s\n' "accepted one-time recomputation at J2=${TARGET_J2}" > "${DECISION_FILE}"
        else
            RETRY_STATUS=$?
            [[ "${RETRY_STATUS}" == "10" ]] || exit "${RETRY_STATUS}"
            cp -f -- "${INITIAL_INPUT}" "${NEXT_CKPT}"
            printf '%s\n' "recomputation at J2=${TARGET_J2} also has |eta| < ${ETA_THRESHOLD}; next J2 uses the pre-attempt tensor" > "${DECISION_FILE}"
        fi
    fi
fi

[[ -s "${NEXT_CKPT}" && -s "${DECISION_FILE}" ]] || {
    echo "continuation guard failed to create its next-stage tensor" >&2; exit 6;
}
cat "${DECISION_FILE}"
echo "COMPLETED ${ALIAS} ${DIRECTION} stage ${STAGE_INDEX}, J2=${TARGET_J2}"
