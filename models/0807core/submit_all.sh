#!/bin/bash

# Submit all fresh TwoC3 jobs requested for D=9, 10, and 11.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_SCRIPT_DIR="/scratch/pghosh/Working_AD_Honeycomb_August07"
D9_J2_VALUES=(0.28 0.30 0.31 0.32 0.33 0.34)
D10_J2_VALUES=(0.265 0.275 0.30 0.31 0.32 0.33 0.34)
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
    local J2_SHORT="${J2#0.}"

    if sbatch \
        --chdir="${SCRIPT_DIR}" \
        --job-name="${J2_SHORT}twoD${D}" \
        --export="ALL,D=${D},J2=${J2}" \
        "${SCRIPT_DIR}/singleFileSbatchTwoC3.run"; then
        submitted=$((submitted + 1))
    else
        failures=$((failures + 1))
    fi
}

for J2 in "${D9_J2_VALUES[@]}"; do
    submit_job 9 "${J2}"
done

for J2 in "${D10_J2_VALUES[@]}"; do
    submit_job 10 "${J2}"
done

for J2 in "${D11_J2_VALUES[@]}"; do
    submit_job 11 "${J2}"
done

if (( failures > 0 )); then
    echo "Submitted ${submitted} jobs; ${failures} sbatch submission(s) failed." >&2
    exit 1
fi

echo "Submitted all ${submitted} fresh TwoC3 jobs."
