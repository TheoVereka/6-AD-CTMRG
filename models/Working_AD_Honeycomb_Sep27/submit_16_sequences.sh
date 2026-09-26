#!/usr/bin/env bash
# One-command launcher: validate/materialise four seeds and submit 16 chains.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_16_sequences.sh [--dry-run]" >&2
    exit 2
fi

EXPECTED_ROOT="${SEP27_EXPECTED_ROOT:-/scratch/pghosh/Working_AD_Honeycomb_Sep27}"
if [[ "${BUNDLE_DIR}" != "${EXPECTED_ROOT}" ]]; then
    echo "Copy this whole folder to ${EXPECTED_ROOT}; got ${BUNDLE_DIR}" >&2
    exit 3
fi
for required in main_C3_LBFGS.py core_C3.py single_stage.run run_one_stage.sh \
        prepare_selection.py candidate_catalog.tsv selection.tsv; do
    [[ -f "${BUNDLE_DIR}/${required}" ]] || { echo "missing ${required}" >&2; exit 3; }
done

# This is deliberately automatic: after the folder is copied to /scratch,
# this single bash command validates the bundled candidates and makes the plan.
python3 "${BUNDLE_DIR}/prepare_selection.py"
PLAN="${BUNDLE_DIR}/submission_plan.tsv"
mkdir -p "${BUNDLE_DIR}/slurm_logs" "${BUNDLE_DIR}/Results_Sep27"

if [[ "${DRY_RUN}" == "0" ]]; then
    mapfile -t existing < <(
        squeue --noheader --user "${USER}" --format '%A %j' 2>/dev/null |
        awk '$2 ~ /^D10a0[1-4][LR][12]s[0-9][0-9]$/ {print $1 ":" $2}'
    )
    if (( ${#existing[@]} > 0 )); then
        echo "Refusing duplicate submission; matching jobs already exist:" >&2
        printf '  %s\n' "${existing[@]}" >&2
        exit 6
    fi
fi

chains=0
stage_jobs=0
dependent_jobs=0
while IFS=$'\t' read -r ALIAS CANDIDATE_ID D CHI SEED_J2 TEXTURE ORIENTATION \
        INSURANCE DIRECTION J2_SEQUENCE STAGE_HOURS; do
    [[ "${ALIAS}" == "alias" ]] && continue
    [[ -n "${ALIAS}" ]] || continue
    chains=$((chains + 1))
    PREVIOUS_JOB_ID=""
    PREVIOUS_CKPT="${BUNDLE_DIR}/seeds/${ALIAS}/tensor_best.pt"
    [[ -s "${PREVIOUS_CKPT}" ]] || { echo "missing seed ${ALIAS}" >&2; exit 3; }
    SHORT_DIRECTION=L
    [[ "${DIRECTION}" == "right" ]] && SHORT_DIRECTION=R
    IFS=: read -r -a TARGETS <<< "${J2_SEQUENCE}"
    STAGE_INDEX=0
    for TARGET_J2 in "${TARGETS[@]}"; do
        STAGE_INDEX=$((STAGE_INDEX + 1))
        stage_jobs=$((stage_jobs + 1))
        TARGET_TAG="${TARGET_J2/./p}"
        OUTPUT_DIR="${BUNDLE_DIR}/Results_Sep27/${ALIAS}/${DIRECTION}/insurance_${INSURANCE}/J2_${TARGET_TAG}"
        CURRENT_BEST="${OUTPUT_DIR}/sweep_D${D}_chi${CHI}_best.pt"
        JOB_NAME="D10${ALIAS}${SHORT_DIRECTION}${INSURANCE}s$(printf '%02d' "${STAGE_INDEX}")"
        DEPENDENCY_TEXT="none"
        if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
            DEPENDENCY_TEXT="afterok:${PREVIOUS_JOB_ID}"
            dependent_jobs=$((dependent_jobs + 1))
        fi
        printf '%02d  %-13s alias=%s copy=%s J2=%-5s %-5s stage=%d dependency=%s\n' \
            "${stage_jobs}" "${JOB_NAME}" "${ALIAS}" "${INSURANCE}" "${TARGET_J2}" \
            "${DIRECTION}" "${STAGE_INDEX}" "${DEPENDENCY_TEXT}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            CURRENT_JOB_ID="DRY${stage_jobs}"
        else
            EXPORTS="ALL,BUNDLE_DIR=${BUNDLE_DIR},ALIAS=${ALIAS},D=${D},CHI=${CHI},SEED_J2=${SEED_J2},TARGET_J2=${TARGET_J2},TEXTURE=${TEXTURE},ORIENTATION=${ORIENTATION},INSURANCE=${INSURANCE},DIRECTION=${DIRECTION},STAGE_INDEX=${STAGE_INDEX},STAGE_HOURS=${STAGE_HOURS},PREVIOUS_CKPT=${PREVIOUS_CKPT},OUTPUT_DIR=${OUTPUT_DIR}"
            SBATCH_ARGS=(--parsable --chdir="${BUNDLE_DIR}" --job-name="${JOB_NAME}" --export="${EXPORTS}")
            if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
                SBATCH_ARGS+=(--dependency="afterok:${PREVIOUS_JOB_ID}")
            fi
            SBATCH_RESULT="$(sbatch "${SBATCH_ARGS[@]}" "${BUNDLE_DIR}/single_stage.run")"
            CURRENT_JOB_ID="${SBATCH_RESULT%%;*}"
            [[ "${CURRENT_JOB_ID}" =~ ^[0-9]+$ ]] || {
                echo "could not parse sbatch job id: ${SBATCH_RESULT}" >&2; exit 7;
            }
            echo "     submitted job=${CURRENT_JOB_ID}"
        fi
        PREVIOUS_JOB_ID="${CURRENT_JOB_ID}"
        PREVIOUS_CKPT="${CURRENT_BEST}"
    done
done < "${PLAN}"

[[ "${chains}" == "16" && "${stage_jobs}" == "64" && "${dependent_jobs}" == "48" ]] || {
    echo "plan invariant failed: chains=${chains}, jobs=${stage_jobs}, dependencies=${dependent_jobs}" >&2
    exit 5
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run passed: 16 chain heads + 48 afterok jobs = 64 jobs; nothing submitted."
else
    echo "Submitted 16 chain heads + 48 afterok jobs = 64 jobs."
fi
