#!/usr/bin/env bash
# Submit 16 independent heads (D6/D7 x source F/R x four replicas), each
# followed by ten afterok stages through 100%.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
ORIGINAL_BUNDLE="${ORIGINAL_BUNDLE:-/scratch/izar/chye/connectionPlaqDimer_trees_20261003}"
MANIFEST="${BUNDLE_DIR}/submitted_plaq_refinement_20261006.tsv"

DRY_RUN=0
case "${1:-}" in
    --dry-run) [[ $# -eq 1 ]] || { echo 'Usage: bash submit_plaq_refinement.sh [--dry-run]' >&2; exit 2; }; DRY_RUN=1 ;;
    '') [[ $# -eq 0 ]] || { echo 'Usage: bash submit_plaq_refinement.sh [--dry-run]' >&2; exit 2; } ;;
    *) echo 'Usage: bash submit_plaq_refinement.sh [--dry-run]' >&2; exit 2 ;;
esac

for file in core_C3.py main_C3_LBFGS.py run_plaq_refinement.sh plaq_refinement_3days.run; do
    [[ -s "${BUNDLE_DIR}/${file}" ]] || { echo "Missing ${file}" >&2; exit 3; }
done
if [[ "${DRY_RUN}" == 0 ]]; then
    command -v sbatch >/dev/null || { echo 'sbatch is unavailable' >&2; exit 3; }
    [[ ! -e "${MANIFEST}" ]] || {
        echo "Manifest already exists: ${MANIFEST}; inspect it before resubmitting." >&2
        exit 4
    }
    # Check every F/R starting point before submitting the first job.
    ORIGINAL_BUNDLE="${ORIGINAL_BUNDLE}" bash "${BUNDLE_DIR}/run_plaq_refinement.sh" --check-sources
    mkdir -p "${BUNDLE_DIR}/slurm_logs" "${BUNDLE_DIR}/results_plaq_refinement_20261006"
    printf 'job_id\tD\tsource_node\treplica\tpercent\tseed\tpredecessor_percent\tdependency\n' > "${MANIFEST}"
fi

declare -A HEAD_IDS=()
JOBS=0
HEADS=0
DEPENDENCIES=0
LAST_ID=""

submit_one() {
    local D="$1" source_node="$2" replica="$3" percent="$4" prev_percent="$5" parent_id="$6"
    local source_offset=0 run_seed job_name dependency=none result job_id
    [[ "${source_node}" == R ]] && source_offset=50
    run_seed=$((1000000 + D * 10000 + source_offset + 10#${replica}))
    job_name="pl${D}${source_node}${replica}p${percent}"
    JOBS=$((JOBS + 1))
    if [[ "${prev_percent}" == none ]]; then
        HEADS=$((HEADS + 1))
    else
        DEPENDENCIES=$((DEPENDENCIES + 1))
        dependency="afterok:${parent_id}"
    fi

    if [[ "${DRY_RUN}" == 1 ]]; then
        job_id="dry${JOBS}"
    else
        local -a options=(
            --parsable
            --chdir="${BUNDLE_DIR}"
            --job-name="${job_name}"
            --export="ALL,D=${D},SOURCE_NODE=${source_node},REPLICA=${replica},PERCENT=${percent},PREV_PERCENT=${prev_percent},RUN_SEED=${run_seed},ORIGINAL_BUNDLE=${ORIGINAL_BUNDLE}"
        )
        if [[ "${prev_percent}" != none ]]; then
            options+=(--dependency="${dependency}")
        fi
        result="$(sbatch "${options[@]}" "${BUNDLE_DIR}/plaq_refinement_3days.run")"
        job_id="${result%%;*}"
        [[ "${job_id}" =~ ^[0-9]+$ ]] || {
            echo "Could not parse sbatch job ID: ${result}" >&2
            exit 5
        }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${job_id}" "${D}" "${source_node}" "${replica}" "${percent}" \
            "${run_seed}" "${prev_percent}" "${dependency}" >> "${MANIFEST}"
    fi
    LAST_ID="${job_id}"
    printf '%03d  D=%s source=%s replica=%s p=%3s%%  job=%-12s dependency=%s\n' \
        "${JOBS}" "${D}" "${source_node}" "${replica}" "${percent}" \
        "${job_id}" "${dependency}"
}

# Submit all sixteen heads first so none waits for the other chains' tails.
for D in 6 7; do
    for source_node in F R; do
        for replica in 01 02 03 04; do
            submit_one "${D}" "${source_node}" "${replica}" 70 none ''
            key="${D}_${source_node}_${replica}"
            HEAD_IDS["${key}"]="${LAST_ID}"
        done
    done
done

for D in 6 7; do
    for source_node in F R; do
        for replica in 01 02 03 04; do
            previous_percent=70
            key="${D}_${source_node}_${replica}"
            previous_id="${HEAD_IDS[${key}]}"
            for percent in 75 80 84 88 91 94 96 98 99 100; do
                submit_one "${D}" "${source_node}" "${replica}" "${percent}" \
                    "${previous_percent}" "${previous_id}"
                previous_percent="${percent}"
                previous_id="${LAST_ID}"
            done
        done
    done
done

[[ "${JOBS}" == 176 && "${HEADS}" == 16 && "${DEPENDENCIES}" == 160 ]] || {
    echo "Internal count error: jobs=${JOBS} heads=${HEADS} dependencies=${DEPENDENCIES}" >&2
    exit 6
}
if [[ "${DRY_RUN}" == 1 ]]; then
    echo 'Dry run: 176 jobs, 16 independent heads, 160 afterok dependencies; nothing submitted.'
else
    echo "Submitted 176 jobs, 16 heads, 160 afterok dependencies. IDs: ${MANIFEST}"
fi
