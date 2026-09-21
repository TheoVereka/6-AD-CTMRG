#!/usr/bin/env bash
# Submit the replica-1 h=0.005 -> 0.003 -> 0.002 -> 0.001 -> 0 Izar sweep.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_izar_small_h.sh [--dry-run]" >&2
    exit 2
fi

for REQUIRED in main_C3_LBFGS.py core_C3.py run_vbc_small_h_izar.sh \
        izar_memory_guard.sh seed_manifest.csv izar_small_h_3days_lbfgs.run \
        izar_small_h_7days_lbfgs.run; do
    [[ -f "${BUNDLE_DIR}/${REQUIRED}" ]] || { echo "Missing ${REQUIRED}" >&2; exit 1; }
done

# Keep exactly the established Lyra-folder/Izar production order.
J2_SEVEN_DAY=(0.275 0.27 0.28 0.265 0.26 0.33 0.34)
J2_THREE_DAY=(0.275 0.27 0.28 0.265 0.26 0.29 0.30 0.31 0.32 0.33 0.34)
BRANCHES=(plaquette dimer-plaquette)
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results_Izar_replica1}"

chi_for_D() {
    case "$1" in
        6) echo 72 ;;
        7) echo 91 ;;
        8) echo 104 ;;
        9) echo 108 ;;
        *) return 1 ;;
    esac
}

seed_for() {
    local J2="$1" D="$2" tag="${1/./p}"
    echo "${BUNDLE_DIR}/seeds/J2_${tag}/D_${D}/tensor_best.pt"
}

orientation_for() {
    local J2="$1" D="$2"
    awk -F, -v j2="${J2}" -v d="${D}" \
        'NR > 1 && $1 == j2 && $2 == d { print $12; found=1; exit }
         END { if (!found) exit 1 }' "${BUNDLE_DIR}/seed_manifest.csv"
}

# Find queued/running jobs by their exact Slurm name.
active_predecessors() {
    local job_name="$1"
    squeue --noheader --user "${USER}" --name "${job_name}" \
        --format "%A" 2>/dev/null | awk '/^[0-9]+$/ { print }'
}

submit_one() {
    local J2="$1" D="$2" BRANCH="$3" RUNFILE="$4" CLASS="$5"
    local CHI ORIENTATION SEED_CKPT SHORT_BRANCH OLD_JOB_NAME JOB_NAME EXPORTS
    local DEPENDENCY_LABEL="none" SBATCH_RESULT
    local -a ACTIVE_IDS=() ACTIVE_SMALL_IDS=()
    CHI="$(chi_for_D "${D}")"
    ORIENTATION="$(orientation_for "${J2}" "${D}")"
    [[ "${ORIENTATION}" =~ ^[012]$ ]] || { echo "Bad orientation" >&2; exit 4; }
    SEED_CKPT="$(seed_for "${J2}" "${D}")"
    [[ -f "${SEED_CKPT}" ]] || { echo "Missing seed ${SEED_CKPT}" >&2; exit 3; }
    SHORT_BRANCH=p
    [[ "${BRANCH}" == "dimer-plaquette" ]] && SHORT_BRANCH=d
    OLD_JOB_NAME="i${J2/./}D${D}${SHORT_BRANCH}r1"
    JOB_NAME="s${J2/./}D${D}${SHORT_BRANCH}r1"
    EXPORTS="ALL,BUNDLE_DIR=${BUNDLE_DIR},OUTROOT=${OUTROOT},D=${D},CHI=${CHI},J2=${J2},BRANCH=${BRANCH},ORIENTATION=${ORIENTATION},REPLICA=1,SEED_CKPT=${SEED_CKPT}"

    if [[ "${DRY_RUN}" == "0" ]]; then
        mapfile -t ACTIVE_SMALL_IDS < <(active_predecessors "${JOB_NAME}")
        if (( ${#ACTIVE_SMALL_IDS[@]} > 0 )); then
            mapfile -t ACTIVE_IDS < <(active_predecessors "${OLD_JOB_NAME}")
            if (( ${#ACTIVE_IDS[@]} > 0 )); then
                if scancel "${ACTIVE_IDS[@]}"; then
                    cancelled_old=$((cancelled_old + ${#ACTIVE_IDS[@]}))
                else
                    echo "WARNING: scancel reported a failure for old jobs: ${ACTIVE_IDS[*]}" >&2
                fi
            fi
            planned=$((planned + 1))
            skipped_active=$((skipped_active + 1))
            printf '%03d  %-4s  J2=%-5s D=%d chi=%-3d %-17s orientation=%s  %-13s already-active=%s old-cancelled=%d\n' \
                "${planned}" "${CLASS}" "${J2}" "${D}" "${CHI}" "${BRANCH}" \
                "${ORIENTATION}" "${JOB_NAME}" "$(IFS=:; echo "${ACTIVE_SMALL_IDS[*]}")" \
                "${#ACTIVE_IDS[@]}"
            return 0
        fi
        mapfile -t ACTIVE_IDS < <(active_predecessors "${OLD_JOB_NAME}")
        if (( ${#ACTIVE_IDS[@]} > 0 )); then
            local joined
            joined="$(IFS=:; echo "${ACTIVE_IDS[*]}")"
            DEPENDENCY_LABEL="afterany:${joined}"
        fi
    else
        DEPENDENCY_LABEL="auto-detect:${OLD_JOB_NAME}"
    fi

    planned=$((planned + 1))
    printf '%03d  %-4s  J2=%-5s D=%d chi=%-3d %-17s orientation=%s  %-13s cancellation-barrier=%s\n' \
        "${planned}" "${CLASS}" "${J2}" "${D}" "${CHI}" "${BRANCH}" \
        "${ORIENTATION}" "${JOB_NAME}" "${DEPENDENCY_LABEL}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        local -a SBATCH_ARGS=(
            --chdir="${BUNDLE_DIR}"
            --job-name="${JOB_NAME}"
            --export="${EXPORTS}"
        )
        if [[ "${DEPENDENCY_LABEL}" != "none" ]]; then
            SBATCH_ARGS+=(--dependency="${DEPENDENCY_LABEL}")
        fi
        # Queue the replacement first.  Only after sbatch succeeds do we
        # cancel its matching large-h predecessor(s).  The afterany barrier
        # prevents concurrent checkpoint I/O during Slurm cancellation; it
        # does not wait for the predecessor to finish its requested physics.
        SBATCH_RESULT="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${BUNDLE_DIR}/${RUNFILE}")"
        echo "  submitted small-h job ${SBATCH_RESULT}"
        submitted_now=$((submitted_now + 1))
        if (( ${#ACTIVE_IDS[@]} > 0 )); then
            if scancel "${ACTIVE_IDS[@]}"; then
                cancelled_old=$((cancelled_old + ${#ACTIVE_IDS[@]}))
                echo "  cancelled large-h job(s): ${ACTIVE_IDS[*]}"
            else
                echo "WARNING: small-h job ${SBATCH_RESULT} was submitted, but scancel reported a failure for: ${ACTIVE_IDS[*]}" >&2
            fi
        fi
    fi
}

planned=0
submitted_now=0
skipped_active=0
cancelled_old=0
for J2 in "${J2_SEVEN_DAY[@]}"; do
    for D in 9; do
        for BRANCH in "${BRANCHES[@]}"; do
            submit_one "${J2}" "${D}" "${BRANCH}" \
                izar_small_h_7days_lbfgs.run 7day
        done
    done
done
for J2 in "${J2_THREE_DAY[@]}"; do
    for D in 8 7 6; do
        for BRANCH in "${BRANCHES[@]}"; do
            submit_one "${J2}" "${D}" "${BRANCH}" \
                izar_small_h_3days_lbfgs.run 3day
        done
    done
done

[[ "${planned}" == "80" ]] || { echo "Internal error: expected 80 jobs" >&2; exit 5; }
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 80 small-h jobs, nothing submitted."
else
    echo "Small-h plan complete: submitted ${submitted_now}; already active ${skipped_active}; cancelled old large-h jobs ${cancelled_old}; total 80."
fi
