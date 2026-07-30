#!/bin/bash

# Submit five fresh TwoC3 D=10 jobs and the two D=10 resume jobs.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
J2_VALUES=(0.245 0.31 0.32 0.33 0.34)
failures=0

submit_job() {
    if ! sbatch --chdir="${SCRIPT_DIR}" "$@"; then
        failures=$((failures + 1))
    fi
}

for J2 in "${J2_VALUES[@]}"; do
    J2_SHORT="${J2#0.}"
    submit_job \
        --job-name="${J2_SHORT}twoD10" \
        --export="ALL,J2=${J2}" \
        "${SCRIPT_DIR}/singleFileSbatchTwoC3.run"
done

# J2=0.26 resumes at D=10, chi=160 and runs only that chi.
submit_job \
    --job-name="26twoD10r" \
    --export="ALL,J2=0.26" \
    "${SCRIPT_DIR}/singleFileSbatchTwoC3Resume.run"

# J2=0.27 resumes at D=10, chi=140.  The normal E/E-lookahead logic decides
# whether the run proceeds to chi=160; no chi override is supplied.
submit_job \
    --job-name="27twoD10r" \
    --export="ALL,J2=0.27" \
    "${SCRIPT_DIR}/singleFileSbatchTwoC3Resume.run"

if (( failures > 0 )); then
    echo "${failures} sbatch submission(s) failed." >&2
    exit 1
fi

echo "Submitted 5 fresh TwoC3 jobs and 2 resumed TwoC3 jobs."
