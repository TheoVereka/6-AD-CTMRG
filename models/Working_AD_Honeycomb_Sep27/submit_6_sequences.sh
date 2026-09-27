#!/bin/bash
# Submit six fully static Kuma chains: 6 heads + 18 afterok jobs.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
        echo "Could not parse job ID from: ${result}" >&2
        exit 1
    }
    JOBS=$((JOBS + 1))
    echo "${name}: ${LAST_JOB_ID}"
}

submit_chain() {
    local predecessor=""
    local name
    for name in "$@"; do
        submit_job "${name}" "${predecessor}"
        predecessor="${LAST_JOB_ID}"
    done
}

submit_chain D10a01Ls01 D10a01Ls02 D10a01Ls03 D10a01Ls04 D10a01Ls05
submit_chain D10a01Rs01 D10a01Rs02 D10a01Rs03
submit_chain D10a02Ls01 D10a02Ls02 D10a02Ls03 D10a02Ls04 D10a02Ls05 D10a02Ls06
submit_chain D10a02Rs01 D10a02Rs02
submit_chain D10a03Ls01 D10a03Ls02 D10a03Ls03 D10a03Ls04 D10a03Ls05 D10a03Ls06
submit_chain D10a03Rs01 D10a03Rs02

[[ "${JOBS}" == "24" && "${DEPENDENCIES}" == "18" ]] || {
    echo "Internal submission-count error: jobs=${JOBS}, dependencies=${DEPENDENCIES}" >&2
    exit 1
}
echo "Submitted 6 chain heads and 18 dependent jobs."
