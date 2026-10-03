#!/usr/bin/env bash
# Submit four connection trees: D6 plaq, D6 dimer, D7 plaq, D7 dimer.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

DRY_RUN=0
case "${1:-}" in
    --dry-run) [[ $# -eq 1 ]] || { echo "Usage: bash submit_all.sh [--dry-run]" >&2; exit 2; }; DRY_RUN=1 ;;
    '') [[ $# -eq 0 ]] || { echo "Usage: bash submit_all.sh [--dry-run]" >&2; exit 2; } ;;
    *) echo "Usage: bash submit_all.sh [--dry-run]" >&2; exit 2 ;;
esac

for file in core_C3.py main_C3.py main_C3_LBFGS.py run_stage.sh d6.run d7.run; do
    [[ -s "${BUNDLE_DIR}/${file}" ]] || { echo "Missing ${file}" >&2; exit 3; }
done

MANIFEST="${BUNDLE_DIR}/submitted_jobs.tsv"
if [[ "${DRY_RUN}" == 0 && -e "${MANIFEST}" ]]; then
    echo "Submission manifest already exists: ${MANIFEST}" >&2
    echo "Inspect it before making a second submission." >&2
    exit 4
fi

if [[ "${DRY_RUN}" == 0 ]]; then
    mkdir -p slurm_logs results
    printf 'job_id\tD\tconnection\tnode\tt\tpredecessor\tdependency\n' > "${MANIFEST}"
fi

JOBS=0
ROOTS=0
DEPENDENCIES=0
LAST_ID=""

submit_one() {
    local D="$1" connection="$2" node="$3" t="$4" predecessor="$5" parent_id="$6"
    local run_file job_name result job_id dependency_text
    if [[ "${D}" == 6 ]]; then run_file="${BUNDLE_DIR}/d6.run"
    else run_file="${BUNDLE_DIR}/d7.run"; fi
    job_name="c${D}${connection:0:1}${node}"
    JOBS=$((JOBS + 1))
    if [[ "${predecessor}" == none ]]; then
        ROOTS=$((ROOTS + 1))
        dependency_text=none
    else
        DEPENDENCIES=$((DEPENDENCIES + 1))
        dependency_text="afterok:${parent_id}"
    fi

    if [[ "${DRY_RUN}" == 1 ]]; then
        job_id="dry${JOBS}"
    else
        local -a options=(
            --parsable
            --chdir="${BUNDLE_DIR}"
            --job-name="${job_name}"
            --export="ALL,D=${D},CONNECTION=${connection},NODE=${node},T=${t},PREV_NODE=${predecessor}"
        )
        if [[ "${predecessor}" != none ]]; then
            options+=(--dependency="${dependency_text}")
        fi
        result="$(sbatch "${options[@]}" "${run_file}")"
        job_id="${result%%;*}"
        [[ "${job_id}" =~ ^[0-9]+$ ]] || {
            echo "Could not parse sbatch job ID: ${result}" >&2
            exit 5
        }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${job_id}" "${D}" "${connection}" "${node}" "${t}" \
            "${predecessor}" "${dependency_text}" >> "${MANIFEST}"
    fi
    LAST_ID="${job_id}"
    printf '%02d  %-5s t=%s  job=%-12s dependency=%s\n' \
        "${JOBS}" "${job_name}" "${t}" "${job_id}" "${dependency_text}"
}

submit_tree() {
    local D="$1" connection="$2" i node previous_node previous_id head_id
    local -a chain=(A B C D E F G H I)
    local -a branches=(O P Q R S T U)

    previous_node=none
    previous_id=""
    for i in "${!chain[@]}"; do
        node="${chain[i]}"
        submit_one "${D}" "${connection}" "${node}" "${i}" "${previous_node}" "${previous_id}"
        previous_node="${node}"
        previous_id="${LAST_ID}"
        if [[ "${node}" == A ]]; then head_id="${LAST_ID}"; fi
    done
    for i in "${!branches[@]}"; do
        node="${branches[i]}"
        submit_one "${D}" "${connection}" "${node}" "$((i + 2))" A "${head_id}"
    done
}

submit_tree 6 plaq
submit_tree 6 dimer
submit_tree 7 plaq
submit_tree 7 dimer

[[ "${JOBS}" == 64 && "${ROOTS}" == 4 && "${DEPENDENCIES}" == 60 ]] || {
    echo "Internal count error: jobs=${JOBS}, roots=${ROOTS}, dependencies=${DEPENDENCIES}" >&2
    exit 6
}
if [[ "${DRY_RUN}" == 1 ]]; then
    echo "Dry run: 64 jobs (4 roots, 60 afterok dependencies); nothing submitted."
else
    echo "Submitted 64 jobs (4 roots, 60 afterok dependencies). IDs: ${MANIFEST}"
fi
