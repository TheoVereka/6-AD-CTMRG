#!/bin/bash

# Submit the seven TwoC3 D=10 jobs for J2 = 0.24, 0.25, ..., 0.30.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
J2_VALUES=(0.24 0.25 0.26 0.27 0.28 0.29 0.30)
failures=0

for J2 in "${J2_VALUES[@]}"; do
    J2_SHORT="${J2#0.}"
    if ! sbatch \
        --chdir="${SCRIPT_DIR}" \
        --job-name="${J2_SHORT}c3D10" \
        --export="ALL,J2=${J2},D=10" \
        "${SCRIPT_DIR}/singleFileSbatchTwoC3.run"; then
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "${failures} sbatch submission(s) failed." >&2
    exit 1
fi

echo "Submitted all 7 TwoC3 D=10 jobs."
