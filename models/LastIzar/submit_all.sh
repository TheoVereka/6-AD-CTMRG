#!/usr/bin/env bash
# Submit the 33 requested heads and their 38 afterok dependants (71 jobs total).
# All numerical and Slurm settings live in static jobs/*.run files.
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

for REQUIRED in main_C3.py main_C3_LBFGS.py core_C3.py run_stage.sh \
        seeds/D6_J2_0p265_tensor_best.pt; do
    [[ -s "${BUNDLE_DIR}/${REQUIRED}" ]] || {
        echo "Missing required bundle file: ${REQUIRED}" >&2
        exit 3
    }
done
mkdir -p slurm_logs Results_LastIzar

job_count=0
head_count=0
dependency_count=0
LAST_JOB_ID=""

submit_one() {
    local run_name="$1" dependency_id="${2:-}" result job_id
    local run_path="${BUNDLE_DIR}/jobs/${run_name}.run"
    [[ -s "${run_path}" ]] || { echo "Missing ${run_path}" >&2; exit 3; }
    job_count=$((job_count + 1))
    if [[ -z "${dependency_id}" ]]; then
        head_count=$((head_count + 1))
        printf '%03d HEAD     %s\n' "${job_count}" "${run_name}"
    else
        dependency_count=$((dependency_count + 1))
        printf '%03d AFTEROK  %-24s <- %s\n' "${job_count}" "${run_name}" "${dependency_id}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        job_id="dry${job_count}"
    elif [[ -z "${dependency_id}" ]]; then
        result="$(sbatch --parsable --chdir="${BUNDLE_DIR}" "${run_path}")"
        job_id="${result%%;*}"
    else
        result="$(sbatch --parsable --chdir="${BUNDLE_DIR}" \
            --dependency="afterok:${dependency_id}" "${run_path}")"
        job_id="${result%%;*}"
    fi
    [[ -n "${job_id}" ]] || { echo "Could not parse sbatch job ID" >&2; exit 5; }
    LAST_JOB_ID="${job_id}"
}

submit_chain() {
    local dependency="" run_name
    for run_name in "$@"; do
        submit_one "${run_name}" "${dependency}"
        dependency="${LAST_JOB_ID}"
    done
}

submit_pair() {
    local first="$1" second="$2" head_id
    submit_one "${first}" ""
    head_id="${LAST_JOB_ID}"
    submit_one "${second}" "${head_id}"
}

# Task 1: one right-moving D=6 adiabatic chain seeded at J2=0.265.
submit_chain \
    t1_d6_j270 t1_d6_j275 t1_d6_j280 t1_d6_j290 \
    t1_d6_j300 t1_d6_j310 t1_d6_j320

# Task 2: D=6, Adam -> LBFGS at h=.005, then pure-LBFGS h=0.
for code in 320 310 300 290 280 275 270; do
    submit_pair "t2_d6_j${code}_h005" "t2_d6_j${code}_h0"
done

# Task 3: D=6, pure LBFGS at both h=.005 and h=0.
for code in 320 310 300 290 280 275 270; do
    submit_pair "t3_d6_j${code}_h005" "t3_d6_j${code}_h0"
done

# Task 4: D=5, Adam -> LBFGS at h=.005, then pure-LBFGS h=0.
for code in 260 265 270 275 280 290 300 310 320; do
    submit_pair "t4_d5_j${code}_h005" "t4_d5_j${code}_h0"
done

# Task 5: D=5, pure LBFGS at both h=.005 and h=0.
for code in 260 265 270 275 280 290 300 310 320; do
    submit_pair "t5_d5_j${code}_h005" "t5_d5_j${code}_h0"
done

[[ "${job_count}" == "71" ]] || { echo "Expected 71 jobs, got ${job_count}" >&2; exit 6; }
[[ "${head_count}" == "33" ]] || { echo "Expected 33 heads, got ${head_count}" >&2; exit 6; }
[[ "${dependency_count}" == "38" ]] || { echo "Expected 38 dependencies, got ${dependency_count}" >&2; exit 6; }
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 71 jobs = 33 heads + 38 afterok jobs; nothing submitted."
else
    echo "Submitted 71 jobs = 33 heads + 38 afterok jobs."
fi
