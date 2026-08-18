#!/bin/bash

# Submit all fresh TwoC3 jobs requested for the August 19 Kuma run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_SCRIPT_DIR="/scratch/pghosh/Working_AD_Honeycomb_August19"
D11_J2_VALUES=(0.275 0.28 0.29 0.30 0.31 0.32 0.33 0.34)
failures=0
submitted=0

if [[ "${SCRIPT_DIR}" != "${EXPECTED_SCRIPT_DIR}" ]]; then
    echo "Run this script from its scratch copy: ${EXPECTED_SCRIPT_DIR}/submit_all.sh" >&2
    exit 1
fi

for REQUIRED_FILE in main_C3.py core_C3.py singleFileSbatchTwoC3.run; do
    if [[ ! -f "${SCRIPT_DIR}/${REQUIRED_FILE}" ]]; then
        echo "Missing required file: ${SCRIPT_DIR}/${REQUIRED_FILE}" >&2
        exit 1
    fi
done

submit_job() {
    local D="$1"
    local J2="$2"
    local CHI_OVERRIDE="${3:-}"
    local CTM_MAX_STEPS_OVERRIDE="${4:-}"
    local J2_SHORT="${J2#0.}"

    if sbatch \
        --chdir="${SCRIPT_DIR}" \
        --job-name="${J2_SHORT}twoD${D}" \
        --export="ALL,D=${D},J2=${J2},CHI_OVERRIDE=${CHI_OVERRIDE},CTM_MAX_STEPS_OVERRIDE=${CTM_MAX_STEPS_OVERRIDE}" \
        "${SCRIPT_DIR}/singleFileSbatchTwoC3.run"; then
        submitted=$((submitted + 1))
    else
        failures=$((failures + 1))
    fi
}

# Normal settings.
for D in 5 6 7 8 9 10; do
    submit_job "${D}" 0.26
done
for J2 in 0.265 0.275; do
    for D in 8 9 10; do
        submit_job "${D}" "${J2}"
    done
done

# A single chi=132 and at most 13 CTMRG steps.
for J2 in "${D11_J2_VALUES[@]}"; do
    submit_job 11 "${J2}" 132 13
done

# J2=0: one requested chi per D and at most 13 CTMRG steps.
submit_job 8  0 200 13
submit_job 9  0 200 13
submit_job 10 0 170 13

if (( failures > 0 )); then
    echo "Submitted ${submitted} jobs; ${failures} sbatch submission(s) failed." >&2
    exit 1
fi

echo "Submitted all ${submitted} fresh TwoC3 jobs."
