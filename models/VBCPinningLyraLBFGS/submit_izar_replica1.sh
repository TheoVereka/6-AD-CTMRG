#!/usr/bin/env bash
# Submit the requested replica-1 Izar sweep in explicit scientific priority order.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_izar_replica1.sh [--dry-run]" >&2
    exit 2
fi

for REQUIRED in main_C3_LBFGS.py core_C3.py run_vbc_branch_izar.sh \
        izar_memory_guard.sh seed_manifest.csv izar_3days_lbfgs.run \
        izar_7days_lbfgs.run; do
    [[ -f "${BUNDLE_DIR}/${REQUIRED}" ]] || { echo "Missing ${REQUIRED}" >&2; exit 1; }
done

# D9 seven-day jobs are submitted first, in exactly this requested J2 order.
J2_SEVEN_DAY=(0.275 0.27 0.28 0.265 0.26 0.33 0.34)
# D8/D7/D6 use the requested priority prefix, then the 0.29:0.01:0.34 grid.
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

submit_one() {
    local J2="$1" D="$2" BRANCH="$3" RUNFILE="$4" CLASS="$5"
    local CHI ORIENTATION SEED_CKPT SHORT_BRANCH JOB_NAME EXPORTS
    CHI="$(chi_for_D "${D}")"
    ORIENTATION="$(orientation_for "${J2}" "${D}")"
    [[ "${ORIENTATION}" =~ ^[012]$ ]] || { echo "Bad orientation" >&2; exit 4; }
    SEED_CKPT="$(seed_for "${J2}" "${D}")"
    [[ -f "${SEED_CKPT}" ]] || { echo "Missing seed ${SEED_CKPT}" >&2; exit 3; }
    SHORT_BRANCH=p
    [[ "${BRANCH}" == "dimer-plaquette" ]] && SHORT_BRANCH=d
    JOB_NAME="i${J2/./}D${D}${SHORT_BRANCH}r1"
    EXPORTS="ALL,BUNDLE_DIR=${BUNDLE_DIR},OUTROOT=${OUTROOT},D=${D},CHI=${CHI},J2=${J2},BRANCH=${BRANCH},ORIENTATION=${ORIENTATION},REPLICA=1,SEED_CKPT=${SEED_CKPT}"

    submitted=$((submitted + 1))
    printf '%03d  %-4s  J2=%-5s D=%d chi=%-3d %-17s orientation=%s  %s\n' \
        "${submitted}" "${CLASS}" "${J2}" "${D}" "${CHI}" "${BRANCH}" \
        "${ORIENTATION}" "${JOB_NAME}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        sbatch --chdir="${BUNDLE_DIR}" --job-name="${JOB_NAME}" \
            --export="${EXPORTS}" "${BUNDLE_DIR}/${RUNFILE}"
    fi
}

submitted=0
for J2 in "${J2_SEVEN_DAY[@]}"; do
    for D in 9; do
        for BRANCH in "${BRANCHES[@]}"; do
            submit_one "${J2}" "${D}" "${BRANCH}" izar_7days_lbfgs.run 7day
        done
    done
done
for J2 in "${J2_THREE_DAY[@]}"; do
    for D in 8 7 6; do
        for BRANCH in "${BRANCHES[@]}"; do
            submit_one "${J2}" "${D}" "${BRANCH}" izar_3days_lbfgs.run 3day
        done
    done
done

[[ "${submitted}" == "80" ]] || { echo "Internal error: expected 80 jobs" >&2; exit 5; }
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 80 jobs, nothing submitted."
else
    echo "Submitted 80 replica-1 jobs in the printed order."
fi
