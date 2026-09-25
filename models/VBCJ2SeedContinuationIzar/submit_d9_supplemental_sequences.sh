#!/usr/bin/env bash
# Submit three supplemental D9 seeds as 12 independent dependency-chain heads.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_d9_supplemental_sequences.sh [--dry-run]" >&2
    exit 2
fi

PLAN="${BUNDLE_DIR}/d9_supplemental_submission_plan.tsv"
CODE_DIR="${CODE_DIR:-/home/chye/VBCPinningLyraLBFGS}"
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results_Izar_J2_sequences}"
for REQUIRED in d9_supplemental_submission_plan.tsv \
        d9_supplemental_seed_manifest.csv run_one_j2_stage_izar.sh \
        izar_3days_sequence.run; do
    [[ -f "${BUNDLE_DIR}/${REQUIRED}" ]] || {
        echo "Missing ${BUNDLE_DIR}/${REQUIRED}" >&2
        exit 3
    }
done
for REQUIRED in main_C3_LBFGS.py core_C3.py izar_memory_guard.sh; do
    [[ -f "${CODE_DIR}/${REQUIRED}" ]] || {
        echo "Missing numerical code: ${CODE_DIR}/${REQUIRED}" >&2
        exit 3
    }
done
mkdir -p "${BUNDLE_DIR}/slurm_logs"

# Existing vjc jobs are deliberately allowed to continue.  Refuse only a
# second copy of this supplemental vjd workflow.
if [[ "${DRY_RUN}" == "0" ]]; then
    mapfile -t EXISTING < <(
        squeue --noheader --user "${USER}" --format '%A %j' 2>/dev/null |
        awk '$2 ~ /^vjd10[123][lr][12]s[0-9][0-9]$/ {print $1 ":" $2}'
    )
    if (( ${#EXISTING[@]} > 0 )); then
        echo "Refusing duplicate supplemental submission:" >&2
        printf '  %s\n' "${EXISTING[@]}" >&2
        exit 6
    fi
fi

planned_stages=0
root_jobs=0
dependent_jobs=0
while IFS=$'\t' read -r SEED_ID D CHI SEED_J2 TEXTURE ORIENTATION \
        DIRECTION J2_SEQUENCE STAGE_HOURS LAUNCHER; do
    [[ "${SEED_ID}" == "seed_id" ]] && continue
    [[ -n "${SEED_ID}" ]] || continue
    [[ "${D}" == "9" && "${CHI}" == "108" ]] || {
        echo "Supplemental plan must be D9/chi108" >&2; exit 4;
    }
    [[ "${LAUNCHER}" == "izar_3days_sequence.run" ]] || {
        echo "Every supplemental stage must use the three-day launcher" >&2; exit 4;
    }
    SEED_CKPT="${BUNDLE_DIR}/d9_supplemental_seeds/${SEED_ID}/tensor_best.pt"
    [[ -s "${SEED_CKPT}" ]] || { echo "Missing ${SEED_CKPT}" >&2; exit 3; }
    SHORT_DIRECTION=l
    [[ "${DIRECTION}" == "right" ]] && SHORT_DIRECTION=r
    IFS=: read -r -a TARGETS <<< "${J2_SEQUENCE}"

    for INSURANCE in 1 2; do
        PREVIOUS_JOB_ID=""
        PREVIOUS_CKPT="${SEED_CKPT}"
        STAGE_INDEX=0
        RUNROOT="${OUTROOT}/${SEED_ID}_J2_${SEED_J2/./p}_D_${D}_${TEXTURE}/${DIRECTION}/insurance_${INSURANCE}"
        for TARGET_J2 in "${TARGETS[@]}"; do
            STAGE_INDEX=$((STAGE_INDEX + 1))
            planned_stages=$((planned_stages + 1))
            if [[ "${STAGE_INDEX}" == "1" ]]; then
                root_jobs=$((root_jobs + 1))
                DEPENDENCY_TEXT="none (chain head)"
            else
                dependent_jobs=$((dependent_jobs + 1))
                DEPENDENCY_TEXT="afterok:${PREVIOUS_JOB_ID}"
            fi
            TARGET_TAG="${TARGET_J2/./p}"
            OUTPUT_DIR="${RUNROOT}/J2_${TARGET_TAG}"
            CURRENT_BEST="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
            JOB_NAME="vjd${SEED_ID#s}${SHORT_DIRECTION}${INSURANCE}s$(printf '%02d' "${STAGE_INDEX}")"
            printf '%03d  %-13s J2=%-5s seed=%s copy=%d %-5s stage=%d dependency=%s\n' \
                "${planned_stages}" "${JOB_NAME}" "${TARGET_J2}" "${SEED_ID}" \
                "${INSURANCE}" "${DIRECTION}" "${STAGE_INDEX}" "${DEPENDENCY_TEXT}"

            if [[ "${DRY_RUN}" == "1" ]]; then
                CURRENT_JOB_ID="DRY-${JOB_NAME}"
            else
                EXPORTS="ALL,BUNDLE_DIR=${BUNDLE_DIR},CODE_DIR=${CODE_DIR},SEED_ID=${SEED_ID},D=${D},CHI=${CHI},SEED_J2=${SEED_J2},TARGET_J2=${TARGET_J2},TEXTURE=${TEXTURE},ORIENTATION=${ORIENTATION},DIRECTION=${DIRECTION},INSURANCE=${INSURANCE},STAGE_INDEX=${STAGE_INDEX},STAGE_HOURS=${STAGE_HOURS},PREVIOUS_CKPT=${PREVIOUS_CKPT},OUTPUT_DIR=${OUTPUT_DIR}"
                SBATCH_ARGS=(--parsable --chdir="${BUNDLE_DIR}" \
                    --job-name="${JOB_NAME}" --export="${EXPORTS}")
                if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
                    SBATCH_ARGS+=(--dependency="afterok:${PREVIOUS_JOB_ID}")
                fi
                SBATCH_RESULT="$(sbatch "${SBATCH_ARGS[@]}" "${BUNDLE_DIR}/${LAUNCHER}")"
                CURRENT_JOB_ID="${SBATCH_RESULT%%;*}"
                [[ "${CURRENT_JOB_ID}" =~ ^[0-9]+$ ]] || {
                    echo "Could not parse sbatch job id: ${SBATCH_RESULT}" >&2; exit 7;
                }
                echo "     submitted job=${CURRENT_JOB_ID}"
            fi
            PREVIOUS_JOB_ID="${CURRENT_JOB_ID}"
            PREVIOUS_CKPT="${CURRENT_BEST}"
        done
    done
done < "${PLAN}"

[[ "${planned_stages}" == "48" ]] || {
    echo "Expected 48 stage jobs, got ${planned_stages}" >&2; exit 5;
}
[[ "${root_jobs}" == "12" && "${dependent_jobs}" == "36" ]] || {
    echo "Expected 12 heads + 36 dependents; got ${root_jobs} + ${dependent_jobs}" >&2; exit 5;
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: 48 three-day D9 jobs = 12 heads + 36 afterok dependents."
else
    echo "Submitted 48 three-day D9 jobs: 12 heads + 36 afterok dependents."
fi
