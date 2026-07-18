#!/bin/bash

# Submit 7 TwoC3 D=9 jobs, 7 resumed C6 D=9 jobs, and the unchanged AD 0.275 job.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
J2_VALUES=(0.24 0.25 0.26 0.27 0.28 0.29 0.30)
failures=0

submit_job() {
    if ! sbatch --chdir="${SCRIPT_DIR}" "$@"; then
        failures=$((failures + 1))
    fi
}

for J2 in "${J2_VALUES[@]}"; do
    J2_SHORT="${J2#0.}"
    submit_job \
        --job-name="${J2_SHORT}twoC3D9" \
        --export="ALL,J2=${J2}" \
        "${SCRIPT_DIR}/singleFileSbatchTwoC3.run"
done

for J2 in "${J2_VALUES[@]}"; do
    J2_SHORT="${J2#0.}"
    submit_job \
        --job-name="${J2_SHORT}c6D9r" \
        --export="ALL,J2=${J2}" \
        "${SCRIPT_DIR}/singleFileSbatchC6resume.run"
done

# This file is intentionally submitted without any overrides or edits.
submit_job "${SCRIPT_DIR}/running_AD_275.run"

if (( failures > 0 )); then
    echo "${failures} sbatch submission(s) failed." >&2
    exit 1
fi

echo "Submitted all 15 jobs."
