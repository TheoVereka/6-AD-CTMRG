#!/usr/bin/env bash
# Submit three static Kuma chains: 3 heads + 3 afterok h=0 jobs.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${BUNDLE_DIR}/logs"
LAST_JOB_ID=""
JOBS=0
DEPENDENCIES=0

submit_job() {
    local name="$1"
    local predecessor="${2:-}"
    local run_file="${BUNDLE_DIR}/jobs/${name}.run"
    local result
    [[ -f "${run_file}" ]] || { echo "Missing ${run_file}" >&2; exit 1; }
    if [[ -n "${predecessor}" ]]; then
        result="$(sbatch --chdir="${BUNDLE_DIR}" \
            --dependency="afterok:${predecessor}" "${run_file}")"
        DEPENDENCIES=$((DEPENDENCIES + 1))
    else
        result="$(sbatch --chdir="${BUNDLE_DIR}" "${run_file}")"
    fi
    LAST_JOB_ID="$(awk '/Submitted batch job [0-9]+/ {print $4}' <<< "${result}" | tail -n 1)"
    [[ "${LAST_JOB_ID}" =~ ^[0-9]+$ ]] || {
        echo "Could not parse job ID from: ${result}" >&2; exit 1;
    }
    JOBS=$((JOBS + 1))
    echo "${name}: ${LAST_JOB_ID}"
}

submit_pair() {
    local pin="$1"
    local zero="$2"
    submit_job "${pin}"
    local head="${LAST_JOB_ID}"
    submit_job "${zero}" "${head}"
}

submit_pair r01_pin r01_zero
submit_pair r02_pin r02_zero
submit_pair r03_pin r03_zero

[[ "${JOBS}" == "6" && "${DEPENDENCIES}" == "3" ]] || {
    echo "Internal submission-count error: jobs=${JOBS}, dependencies=${DEPENDENCIES}" >&2
    exit 1
}
echo "Submitted 3 chain heads and 3 afterok jobs."
