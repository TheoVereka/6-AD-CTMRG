#!/usr/bin/env bash
# Restartable replica-1 small-h continuation for one physical VBC pin.
set -euo pipefail

: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"
: "${J2:?J2 must be exported}"
: "${BRANCH:?BRANCH must be exported}"
: "${ORIENTATION:?ORIENTATION must be exported}"
: "${SEED_CKPT:?SEED_CKPT must point to a bundled 0713summary tensor}"
: "${STAGE_HOURS:?STAGE_HOURS must be exported by the Slurm launcher}"

if [[ "${BRANCH}" != "plaquette" && "${BRANCH}" != "dimer-plaquette" ]]; then
    echo "Only the two physical pins are allowed; got BRANCH=${BRANCH}" >&2
    exit 2
fi
if [[ ! "${ORIENTATION}" =~ ^[012]$ ]]; then
    echo "Invalid geometrical VBC orientation: ${ORIENTATION}" >&2
    exit 2
fi
if [[ ! -f "${SEED_CKPT}" ]]; then
    echo "Missing bundled 0713summary seed: ${SEED_CKPT}" >&2
    exit 3
fi

BUNDLE_DIR="${BUNDLE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BUNDLE_DIR="$(cd -- "${BUNDLE_DIR}" && pwd)"
cd "${BUNDLE_DIR}"

# The order is the physics protocol agreed for retaining the local VBC basin.
# A completed stage is skipped; a partial stage resumes its latest checkpoint.
FIELDS=(0.005 0.003 0.002 0.001 0)
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results_Izar_replica1}"
RUNROOT="${OUTROOT}/J2_${J2}/D_${D}/orientation_${ORIENTATION}/${BRANCH}/replica_1"
mkdir -p "${RUNROOT}"

stage_dir_for() {
    local field="$1"
    echo "${RUNROOT}/h_${field/./p}"
}

# Select the tensor closest to h=0.005 that exists when this job starts.
# For a complete stage use its variational best; for a partial stage use its
# latest checkpoint.  This lets the job follow an older Izar job via a Slurm
# dependency without requiring the old job to have reached every field.
find_start_checkpoint() {
    local field stage best latest observation
    for field in 0.005 0.01 0.02 0.04 0.08; do
        stage="$(stage_dir_for "${field}")"
        best="${stage}/sweep_D${D}_chi${CHI}_best.pt"
        latest="${stage}/sweep_D${D}_chi${CHI}_latest.pt"
        observation="${stage}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
        if [[ -f "${best}" && -f "${observation}" ]]; then
            echo "${best}"
            return 0
        fi
        if [[ -f "${latest}" ]]; then
            echo "${latest}"
            return 0
        fi
        if [[ -f "${best}" ]]; then
            echo "${best}"
            return 0
        fi
    done
    echo "${SEED_CKPT}"
}

J2_CODE="${J2/./}"
J2_INTEGER=$((10#${J2_CODE}))
SEED=$((1000000 + 10000 * D + 100 * ORIENTATION + J2_INTEGER))
PREVIOUS_CKPT="$(find_start_checkpoint)"
echo "initial continuation tensor: ${PREVIOUS_CKPT}"

for FIELD in "${FIELDS[@]}"; do
    STAGE_DIR="$(stage_dir_for "${FIELD}")"
    BEST_CKPT="${STAGE_DIR}/sweep_D${D}_chi${CHI}_best.pt"
    LATEST_CKPT="${STAGE_DIR}/sweep_D${D}_chi${CHI}_latest.pt"
    OBS_FILE="${STAGE_DIR}/D_${D}_chi_${CHI}_energy_magnetization_correlation.txt"
    mkdir -p "${STAGE_DIR}"

    if [[ -f "${BEST_CKPT}" && -f "${OBS_FILE}" ]]; then
        echo "stage h=${FIELD} already complete; continuing from ${BEST_CKPT}"
        PREVIOUS_CKPT="${BEST_CKPT}"
        continue
    fi
    if [[ -f "${LATEST_CKPT}" ]]; then
        echo "stage h=${FIELD} is partial; resuming latest checkpoint"
        PREVIOUS_CKPT="${LATEST_CKPT}"
    elif [[ -f "${BEST_CKPT}" ]]; then
        echo "stage h=${FIELD} has only a best checkpoint; resuming it"
        PREVIOUS_CKPT="${BEST_CKPT}"
    fi
    if [[ ! -f "${PREVIOUS_CKPT}" ]]; then
        echo "Missing continuation checkpoint: ${PREVIOUS_CKPT}" >&2
        exit 3
    fi

    echo "J2=${J2} D=${D} chi=${CHI} branch=${BRANCH} orientation=${ORIENTATION} h=${FIELD} pure-LBFGS noise<=1e-3"
    python "${BUNDLE_DIR}/main_C3_LBFGS.py" \
        --J2 "${J2}" \
        --ansatz twoc3 \
        --Ds "${D}" \
        --chi-min "${CHI}" \
        --chi-max "${CHI}" \
        --chi-step 1 \
        --hours "${STAGE_HOURS}" \
        --optimizer lbfgs \
        --vbc-branch "${BRANCH}" \
        --vbc-orientation "${ORIENTATION}" \
        --vbc-field "${FIELD}" \
        --noise 0.001 \
        --fix-seed \
        --seed "${SEED}" \
        --output-dir "${STAGE_DIR}" \
        --no-mean-field-init \
        --resume "${PREVIOUS_CKPT}" \
        --resume-tensors-only

    if [[ ! -f "${BEST_CKPT}" || ! -f "${OBS_FILE}" ]]; then
        echo "Stage h=${FIELD} did not finish cleanly in ${STAGE_DIR}" >&2
        exit 4
    fi
    PREVIOUS_CKPT="${BEST_CKPT}"
done

echo "END OF IZAR SMALL-H VBC BRANCH: ${RUNROOT}"
