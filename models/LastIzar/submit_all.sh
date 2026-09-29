#!/usr/bin/env bash
# Submit only the replacement LastIzar workload. The already completed
# insurance-1 D=6 plaquette chain is deliberately not resubmitted.
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
        seeds/D6_J2_0p265_tensor_best.pt \
        seeds/D5_J2_0p29_tensor_best.pt \
        seeds/D6_J2_0p275_tensor_best.pt; do
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
        printf '%03d AFTEROK  %-24s <- %s\n' \
            "${job_count}" "${run_name}" "${dependency_id}"
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

# Four new D=6 plaquette chains. Together with the existing direct task1
# chain these are insurances 1--5.
for insurance in 2 3 4 5; do
    submit_chain \
        "p6i${insurance}_j027" "p6i${insurance}_j0275" \
        "p6i${insurance}_j028" "p6i${insurance}_j029" \
        "p6i${insurance}_j030" "p6i${insurance}_j031" \
        "p6i${insurance}_j032"
done

# Dimer adiabatic chains.
submit_chain d5_j030 d5_j031 d5_j032
submit_chain d6_j027 d6_j0265 d6_j026

submit_pin_dimension() {
    local D="$1" field_code="$2" optimizer code
    local -a codes
    if [[ "${D}" == "6" ]]; then
        codes=(032 031 030 029 028 0275 027)
    else
        codes=(026 0265 027 0275 028 029 030 031 032)
    fi
    for optimizer in a l; do
        for code in "${codes[@]}"; do
            submit_pair \
                "p${D}${optimizer}${field_code}_j${code}_pin" \
                "p${D}${optimizer}${field_code}_j${code}_h0"
        done
    done
}

# Requested launch order: h=.02, .03, .01; D=6 before D=5 for every field.
for field_code in 02 03 01; do
    submit_pin_dimension 6 "${field_code}"
    submit_pin_dimension 5 "${field_code}"
done

[[ "${job_count}" == "226" ]] || {
    echo "Expected 226 jobs, got ${job_count}" >&2; exit 6;
}
[[ "${head_count}" == "102" ]] || {
    echo "Expected 102 heads, got ${head_count}" >&2; exit 6;
}
[[ "${dependency_count}" == "124" ]] || {
    echo "Expected 124 dependencies, got ${dependency_count}" >&2; exit 6;
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 226 jobs = 102 heads + 124 afterok jobs; nothing submitted."
else
    echo "Submitted 226 jobs = 102 heads + 124 afterok jobs."
fi
