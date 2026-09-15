#!/bin/bash
# Common branch-following driver used by both Izar and Kuma launchers.
set -euo pipefail

: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${J2:?J2 must be exported}"
: "${BRANCH:?BRANCH must be exported}"
: "${ORIENTATION:?ORIENTATION must be exported}"
: "${REPLICA:?REPLICA must be exported}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported by the launcher}"

if [[ "${BRANCH}" != "plaquette" && "${BRANCH}" != "dimer-plaquette" && "${BRANCH}" != "rank-split" ]]; then
    echo "Invalid BRANCH=${BRANCH}" >&2
    exit 2
fi
if [[ "${BRANCH}" == "rank-split" && ! "${RANK_ORDER:-}" =~ ^[012]:[012]:[012]$ ]]; then
    echo "rank-split requires RANK_ORDER=rank1:rank2:rank3" >&2
    exit 2
fi
if [[ ! "${ORIENTATION}" =~ ^[012]$ ]]; then
    echo "Invalid ORIENTATION=${ORIENTATION}" >&2
    exit 2
fi

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

FIELDS_TEXT="${VBC_FIELDS:-0.08 0.04 0.02 0.01 0.0}"
read -r -a FIELDS <<< "${FIELDS_TEXT}"
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results_VBC_branches}"
RUNROOT="${OUTROOT}/J2_${J2}/D_${D}/orientation_${ORIENTATION}/${BRANCH}/replica_${REPLICA}"
mkdir -p "${RUNROOT}"

J2_CODE="${J2/./}"
SEED=$((100000 * REPLICA + 1000 * D + 10 * ORIENTATION + 10#${J2_CODE}))
PREVIOUS_CKPT="${SEED_CKPT:-}"

for FIELD in "${FIELDS[@]}"; do
    FIELD_LABEL="${FIELD/./p}"
    STAGE_DIR="${RUNROOT}/h_${FIELD_LABEL}"
    BEST_CKPT="${STAGE_DIR}/sweep_D${D}_chi${CHI}_best.pt"
    OBS_FILE="${STAGE_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
    mkdir -p "${STAGE_DIR}"

    # Manual resubmission after a wall-time stop resumes rather than restarting.
    if [[ -f "${BEST_CKPT}" && -f "${OBS_FILE}" ]]; then
        echo "stage h=${FIELD} already complete; continuing from ${BEST_CKPT}"
        PREVIOUS_CKPT="${BEST_CKPT}"
        continue
    fi
    if [[ -f "${BEST_CKPT}" ]]; then
        echo "stage h=${FIELD} has a checkpoint but no observables; resuming it"
        PREVIOUS_CKPT="${BEST_CKPT}"
    fi

    EXTRA_ARGS=(--no-mean-field-init)
    if [[ -n "${PREVIOUS_CKPT}" ]]; then
        if [[ ! -f "${PREVIOUS_CKPT}" ]]; then
            echo "Missing continuation checkpoint: ${PREVIOUS_CKPT}" >&2
            exit 3
        fi
        EXTRA_ARGS+=(--resume "${PREVIOUS_CKPT}" --resume-tensors-only)
    fi

    RANK_ORDER_ARG="${RANK_ORDER:-0:1:2}"
    echo "branch=${BRANCH} orientation=${ORIENTATION} rank_order=${RANK_ORDER_ARG} replica=${REPLICA} h=${FIELD}"
    python "${BUNDLE_DIR}/main_C3.py" \
        --J2 "${J2}" \
        --ansatz twoc3 \
        --Ds "${D}" \
        --chi-min "${CHI}" \
        --chi-max "${CHI}" \
        --chi-step 1 \
        --hours "${STAGE_HOURS}" \
        --vbc-branch "${BRANCH}" \
        --vbc-orientation "${ORIENTATION}" \
        --vbc-rank-order "${RANK_ORDER_ARG}" \
        --vbc-field "${FIELD}" \
        --fix-seed \
        --seed "${SEED}" \
        --output-dir "${STAGE_DIR}" \
        "${EXTRA_ARGS[@]}"

    if [[ ! -f "${BEST_CKPT}" || ! -f "${OBS_FILE}" ]]; then
        echo "Stage h=${FIELD} did not finish cleanly in ${STAGE_DIR}" >&2
        exit 4
    fi
    PREVIOUS_CKPT="${BEST_CKPT}"
done

echo "END OF VBC BRANCH: ${RUNROOT}"
