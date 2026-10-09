#!/usr/bin/env bash
# Recover only the nine D=7 low-J2 stages that are absent from the completed
# snapshot.  The active +h=0.02 J2=0.12 -> 0.10 chain is deliberately omitted.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash submit_missing_D7_7days.sh [--dry-run]" >&2
    exit 2
fi

for required in main_C3_LBFGS.py core_C3.py run_one_stage.sh izar_7days_stage.run; do
    [[ -s "${BUNDLE_DIR}/${required}" ]] || { echo "Missing ${required}" >&2; exit 3; }
done
mkdir -p slurm_logs Results_Izar_lowJ2

job_count=0
head_count=0
dependency_count=0
LAST_JOB_ID=""

submit_one() {
    local job_name="$1" predecessor="$2" exports="$3" result job_id
    job_count=$((job_count + 1))
    if [[ -z "${predecessor}" ]]; then
        head_count=$((head_count + 1))
        printf '%02d HEAD    %s\n' "${job_count}" "${job_name}"
    else
        dependency_count=$((dependency_count + 1))
        printf '%02d AFTEROK %-12s <- %s\n' "${job_count}" "${job_name}" "${predecessor}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        job_id="dry${job_count}"
    else
        local -a args=(
            --parsable
            --chdir="${BUNDLE_DIR}"
            --job-name="${job_name}"
            --export="${exports}"
        )
        [[ -n "${predecessor}" ]] && args+=(--dependency="afterok:${predecessor}")
        result="$(sbatch "${args[@]}" "${BUNDLE_DIR}/izar_7days_stage.run")"
        job_id="${result%%;*}"
    fi
    [[ -n "${job_id}" ]] || { echo "Could not parse sbatch job ID" >&2; exit 5; }
    LAST_JOB_ID="${job_id}"
}

submit_recovery_chain() {
    local texture="$1" signed_h="$2" grid="$3" predecessor_j2="$4"
    shift 4
    local branch sign_code sign_label field field_code
    if [[ "${signed_h}" == -* ]]; then
        branch=dimer-plaquette; sign_code=d; sign_label=m
        [[ "${texture}" == "dimer" ]] || { echo "Texture/sign mismatch" >&2; exit 2; }
    else
        branch=plaquette; sign_code=p; sign_label=p
        [[ "${texture}" == "plaquette" ]] || { echo "Texture/sign mismatch" >&2; exit 2; }
    fi
    field="${signed_h#[-+]}"
    field_code="${field/./p}"
    local predecessor_code="${predecessor_j2/./p}"
    local input_rel="Results_Izar_lowJ2/D_7/${texture}/h_${sign_label}${field_code}/${grid}/J2_${predecessor_code}/sweep_D7_chi126_best.pt"
    if [[ "${DRY_RUN}" != "1" && ! -s "${BUNDLE_DIR}/${input_rel}" ]]; then
        echo "Missing last completed predecessor tensor: ${input_rel}" >&2
        exit 3
    fi

    local predecessor="" j2 j2_code output_rel job_name random_seed exports
    for j2 in "$@"; do
        j2_code="${j2/./p}"
        output_rel="Results_Izar_lowJ2/D_7/${texture}/h_${sign_label}${field_code}/${grid}/J2_${j2_code}"
        job_name="r7${sign_code}${field_code#0p}${grid:0:1}${j2_code#0p}"
        random_seed=$((7900000 + 10000 * job_count + 10#${field#0.} + 10#${j2#0.}))
        exports="ALL,BUNDLE_DIR=${BUNDLE_DIR},D=7,CHI=126,ORIENTATION=1,BRANCH=${branch},SIGNED_H=${signed_h},FIELD=${field},TARGET_J2=${j2},STAGE_HOURS=166,INPUT_REL=${input_rel},OUTPUT_REL=${output_rel},RANDOM_SEED=${random_seed},FORCE_INPUT_CHECKPOINT=1"
        submit_one "${job_name}" "${predecessor}" "${exports}"
        predecessor="${LAST_JOB_ID}"
        input_rel="${output_rel}/sweep_D7_chi126_best.pt"
    done
}

# Last complete J2=0.14; recover 0.12 -> 0.10.
submit_recovery_chain dimer -0.02 main 0.14 0.12 0.10

# Last complete J2=0.16; recover 0.14 -> 0.12 -> 0.10.
submit_recovery_chain dimer -0.04 main 0.16 0.14 0.12 0.10

# Last complete J2=0.08; recover 0.05 -> 0.02.
submit_recovery_chain dimer -0.08 main 0.08 0.05 0.02

# Last complete J2=0.14; recover the final extra-grid point 0.12.
submit_recovery_chain dimer -0.08 extra 0.14 0.12

# Last complete J2=0.12; recover the final +h=0.04 point 0.10.
submit_recovery_chain plaquette +0.04 main 0.12 0.10

[[ "${job_count}" == "9" && "${head_count}" == "5" && "${dependency_count}" == "4" ]] || {
    echo "Internal count error: jobs=${job_count}, heads=${head_count}, dependencies=${dependency_count}" >&2
    exit 6
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 9 jobs = 5 heads + 4 afterok jobs; nothing submitted."
else
    echo "Submitted 9 recovery jobs = 5 heads + 4 afterok jobs, all on seven-day QOS."
fi

