#!/usr/bin/env bash
# Submit four independent adiabatic duplicates and one pin -> h=0 chain.
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_all.sh [--dry-run]" >&2
    exit 2
fi

for required in main_C3.py main_C3_LBFGS.py core_C3.py run_stage.sh \
        seeds/D7_J2_0p265_dimer_tensor_best.pt; do
    [[ -s "${required}" ]] || { echo "Missing ${required}" >&2; exit 3; }
done
mkdir -p slurm_logs Results_D7DimerJ2_0p26
: > submission_job_ids.tsv
printf 'role\tlauncher\tjob_id\tdependency\n' >> submission_job_ids.tsv

submit_one() {
    local role="$1" launcher="$2" dependency="${3:-}" result job_id
    if [[ "${DRY_RUN}" == "1" ]]; then
        job_id="dry_${launcher}"
    elif [[ -n "${dependency}" ]]; then
        result="$(sbatch --parsable --chdir="${BUNDLE_DIR}" \
            --dependency="afterok:${dependency}" "jobs/${launcher}.run")"
        job_id="${result%%;*}"
    else
        result="$(sbatch --parsable --chdir="${BUNDLE_DIR}" "jobs/${launcher}.run")"
        job_id="${result%%;*}"
    fi
    [[ -n "${job_id}" ]] || { echo "Could not parse job ID" >&2; exit 5; }
    printf '%s\t%s\t%s\t%s\n' "${role}" "${launcher}" "${job_id}" "${dependency}" \
        | tee -a submission_job_ids.tsv
    LAST_JOB_ID="${job_id}"
}

for duplicate in 1 2 3 4; do
    submit_one "adiabatic_duplicate_${duplicate}" "d7a026_i${duplicate}"
done
submit_one "pin_h0p02" d7p026_pin
pin_job_id="${LAST_JOB_ID}"
submit_one "pin_to_h0_afterok" d7p026_h0 "${pin_job_id}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: 6 jobs = 5 heads + 1 afterok; nothing submitted."
else
    echo "Submitted 6 jobs = 5 heads + 1 afterok."
fi
