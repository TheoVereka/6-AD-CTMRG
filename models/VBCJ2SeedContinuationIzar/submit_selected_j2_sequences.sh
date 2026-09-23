#!/usr/bin/env bash
# Pre-submit every J2 point as its own Slurm job and link it with afterok.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_selected_j2_sequences.sh [--dry-run]" >&2
    exit 2
fi

PLAN="${BUNDLE_DIR}/submission_plan.tsv"
CODE_DIR="${CODE_DIR:-/home/chye/VBCPinningLyraLBFGS}"
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results_Izar_J2_sequences}"
for REQUIRED in submission_plan.tsv selected_seed_manifest.csv \
        run_one_j2_stage_izar.sh izar_3days_sequence.run izar_7days_sequence.run; do
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

# Refuse accidental double submission of either the obsolete one-allocation
# workflow (cNNN...) or this dependency workflow (vjcNNN...).
if [[ "${DRY_RUN}" == "0" ]]; then
    mapfile -t EXISTING < <(
        squeue --noheader --user "${USER}" --format '%A %j' 2>/dev/null |
        awk '$2 ~ /^c[0-9][0-9][0-9][lr][12]$/ ||
             $2 ~ /^vjc[0-9][0-9][0-9][lr][12]s[0-9][0-9]$/ {print $1 ":" $2}'
    )
    if (( ${#EXISTING[@]} > 0 )); then
        echo "Refusing duplicate submission; workflow jobs are already queued:" >&2
        printf '  %s\n' "${EXISTING[@]}" >&2
        echo "Run bash clear_j2_continuation_izar.sh --yes before a fresh submission." >&2
        exit 6
    fi
fi

planned_stages=0
root_jobs=0
dependent_jobs=0
three_day=0
seven_day=0
while IFS=$'\t' read -r SEED_ID D CHI SEED_J2 TEXTURE ORIENTATION \
        DIRECTION J2_SEQUENCE STAGE_HOURS LAUNCHER; do
    [[ "${SEED_ID}" == "seed_id" ]] && continue
    [[ -n "${SEED_ID}" ]] || continue
    SEED_CKPT="${BUNDLE_DIR}/seeds/${SEED_ID}/tensor_best.pt"
    [[ -s "${SEED_CKPT}" ]] || { echo "Missing ${SEED_CKPT}" >&2; exit 3; }
    [[ "${LAUNCHER}" == "izar_3days_sequence.run" || \
       "${LAUNCHER}" == "izar_7days_sequence.run" ]] || {
        echo "Invalid launcher in plan: ${LAUNCHER}" >&2
        exit 4
    }
    SHORT_DIRECTION=l
    [[ "${DIRECTION}" == "right" ]] && SHORT_DIRECTION=r
    IFS=: read -r -a TARGETS <<< "${J2_SEQUENCE}"
    (( ${#TARGETS[@]} > 0 )) || { echo "Empty sequence for ${SEED_ID}" >&2; exit 4; }

    for INSURANCE in 1 2; do
        PREVIOUS_JOB_ID=""
        PREVIOUS_CKPT="${SEED_CKPT}"
        STAGE_INDEX=0
        RUNROOT="${OUTROOT}/${SEED_ID}_J2_${SEED_J2/./p}_D_${D}_${TEXTURE}/${DIRECTION}/insurance_${INSURANCE}"
        for TARGET_J2 in "${TARGETS[@]}"; do
            STAGE_INDEX=$((STAGE_INDEX + 1))
            planned_stages=$((planned_stages + 1))
            if [[ "${LAUNCHER}" == "izar_3days_sequence.run" ]]; then
                three_day=$((three_day + 1))
            else
                seven_day=$((seven_day + 1))
            fi
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
            JOB_NAME="vjc${SEED_ID#s}${SHORT_DIRECTION}${INSURANCE}s$(printf '%02d' "${STAGE_INDEX}")"
            printf '%03d  %-13s J2=%-5s D=%d copy=%d %-5s stage=%d dependency=%s\n' \
                "${planned_stages}" "${JOB_NAME}" "${TARGET_J2}" "${D}" \
                "${INSURANCE}" "${DIRECTION}" "${STAGE_INDEX}" "${DEPENDENCY_TEXT}"

            if [[ "${DRY_RUN}" == "1" ]]; then
                CURRENT_JOB_ID="DRY-${JOB_NAME}"
            else
                EXPORTS="ALL,BUNDLE_DIR=${BUNDLE_DIR},CODE_DIR=${CODE_DIR},SEED_ID=${SEED_ID},D=${D},CHI=${CHI},SEED_J2=${SEED_J2},TARGET_J2=${TARGET_J2},TEXTURE=${TEXTURE},ORIENTATION=${ORIENTATION},DIRECTION=${DIRECTION},INSURANCE=${INSURANCE},STAGE_INDEX=${STAGE_INDEX},STAGE_HOURS=${STAGE_HOURS},PREVIOUS_CKPT=${PREVIOUS_CKPT},OUTPUT_DIR=${OUTPUT_DIR}"
                SBATCH_ARGS=(
                    --parsable
                    --chdir="${BUNDLE_DIR}"
                    --job-name="${JOB_NAME}"
                    --export="${EXPORTS}"
                )
                if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
                    SBATCH_ARGS+=(--dependency="afterok:${PREVIOUS_JOB_ID}")
                fi
                SBATCH_RESULT="$(sbatch "${SBATCH_ARGS[@]}" "${BUNDLE_DIR}/${LAUNCHER}")"
                CURRENT_JOB_ID="${SBATCH_RESULT%%;*}"
                [[ "${CURRENT_JOB_ID}" =~ ^[0-9]+$ ]] || {
                    echo "Could not parse sbatch job id: ${SBATCH_RESULT}" >&2
                    exit 7
                }
                echo "     submitted job=${CURRENT_JOB_ID}"
            fi
            PREVIOUS_JOB_ID="${CURRENT_JOB_ID}"
            PREVIOUS_CKPT="${CURRENT_BEST}"
        done
    done
done < "${PLAN}"

[[ "${planned_stages}" == "98" ]] || {
    echo "Internal plan error: expected 98 stage jobs, got ${planned_stages}" >&2
    exit 5
}
[[ "${root_jobs}" == "22" && "${dependent_jobs}" == "76" ]] || {
    echo "Internal dependency count error: roots=${root_jobs}, dependent=${dependent_jobs}" >&2
    exit 5
}
[[ "${three_day}" == "32" && "${seven_day}" == "66" ]] || {
    echo "Internal class count error: 3day=${three_day}, 7day=${seven_day}" >&2
    exit 5
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: 98 Slurm jobs = 22 chain heads + 76 afterok-dependent stages."
    echo "Resources: 32 three-day D7 jobs; 66 seven-day D8/D9 jobs. Nothing submitted."
else
    echo "Submitted 98 Slurm jobs: 22 chain heads can queue; 76 wait on afterok dependencies."
fi

