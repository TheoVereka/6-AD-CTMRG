#!/usr/bin/env bash
# Thirty Izar heads and their afterok continuations toward lower J2.
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

for required in main_C3_LBFGS.py core_C3.py run_one_stage.sh \
        izar_3days_stage.run izar_7days_stage.run; do
    [[ -s "${BUNDLE_DIR}/${required}" ]] || { echo "Missing ${required}" >&2; exit 3; }
done
LOCKED_CHI='[ 36, 54, 72, 90,108,126,144,162,160,160,180]'
grep -Fq "CHI_MIN_LIST  = ${LOCKED_CHI}" main_C3_LBFGS.py || {
    echo "Bundled main_C3_LBFGS.py does not have the locked CHI_MIN_LIST" >&2; exit 3;
}
grep -Fq "CHI_MAX_LIST  = ${LOCKED_CHI}" main_C3_LBFGS.py || {
    echo "Bundled main_C3_LBFGS.py does not have the locked CHI_MAX_LIST" >&2; exit 3;
}
for D in 6 7 8; do
    for texture in dimer plaquette; do
        for field_code in 0p01 0p02 0p04 0p08; do
            [[ -s "${BUNDLE_DIR}/seeds/D${D}/${texture}/h_${field_code}.pt" ]] || {
                echo "Missing D=${D} ${texture} h=${field_code} seed" >&2; exit 3;
            }
        done
    done
done
mkdir -p slurm_logs Results_Izar_lowJ2

job_count=0
head_count=0
dependency_count=0
LAST_JOB_ID=""

submit_one() {
    local run_file="$1" job_name="$2" predecessor="$3" exports="$4" result job_id
    job_count=$((job_count + 1))
    if [[ -z "${predecessor}" ]]; then
        head_count=$((head_count + 1))
        printf '%03d HEAD    %-12s %s\n' "${job_count}" "${job_name}" "${run_file}"
    else
        dependency_count=$((dependency_count + 1))
        printf '%03d AFTEROK %-12s <- %s\n' "${job_count}" "${job_name}" "${predecessor}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        job_id="dry${job_count}"
    else
        local -a args=(--parsable --chdir="${BUNDLE_DIR}" --job-name="${job_name}" --export="${exports}")
        [[ -n "${predecessor}" ]] && args+=(--dependency="afterok:${predecessor}")
        result="$(sbatch "${args[@]}" "${BUNDLE_DIR}/${run_file}")"
        job_id="${result%%;*}"
    fi
    [[ -n "${job_id}" ]] || { echo "Could not parse sbatch job ID" >&2; exit 5; }
    LAST_JOB_ID="${job_id}"
}

submit_chain() {
    local D="$1" signed_h="$2" grid="$3" run_file="$4"
    shift 4
    local CHI ORIENTATION texture branch sign_code sign_label field field_code grid_code stage_hours
    case "${D}" in
        6) CHI=108; ORIENTATION=2 ;;
        7) CHI=126; ORIENTATION=1 ;;
        8) CHI=144; ORIENTATION=1 ;;
        *) echo "Unsupported D=${D}" >&2; exit 2 ;;
    esac
    if [[ "${signed_h}" == -* ]]; then
        texture=dimer; branch=dimer-plaquette; sign_code=d; sign_label=m
    else
        texture=plaquette; branch=plaquette; sign_code=p; sign_label=p
    fi
    field="${signed_h#[-+]}"
    field_code="${field/./p}"
    [[ "${grid}" == "main" ]] && grid_code=m || grid_code=x
    [[ "${run_file}" == "izar_3days_stage.run" ]] && stage_hours=70.5 || stage_hours=166

    local predecessor="" input_rel="seeds/D${D}/${texture}/h_${field_code}.pt"
    local j2 j2_code output_rel job_name sign_seed grid_seed random_seed exports
    [[ "${sign_code}" == "p" ]] && sign_seed=1 || sign_seed=0
    [[ "${grid_code}" == "x" ]] && grid_seed=1 || grid_seed=0
    for j2 in "$@"; do
        j2_code="${j2/./p}"
        output_rel="Results_Izar_lowJ2/D_${D}/${texture}/h_${sign_label}${field_code}/${grid}/J2_${j2_code}"
        job_name="i${D}${sign_code}${field_code#0p}${grid_code}${j2_code#0p}"
        random_seed=$((7600000 + 10000 * D + 3000 * sign_seed + 1500 * grid_seed + 10#${field#0.} + 10#${j2#0.}))
        exports="ALL,BUNDLE_DIR=${BUNDLE_DIR},D=${D},CHI=${CHI},ORIENTATION=${ORIENTATION},BRANCH=${branch},SIGNED_H=${signed_h},FIELD=${field},TARGET_J2=${j2},STAGE_HOURS=${stage_hours},INPUT_REL=${input_rel},OUTPUT_REL=${output_rel},RANDOM_SEED=${random_seed}"
        submit_one "${run_file}" "${job_name}" "${predecessor}" "${exports}"
        predecessor="${LAST_JOB_ID}"
        input_rel="${output_rel}/sweep_D${D}_chi${CHI}_best.pt"
    done
}

for D in 6 7 8; do
    if [[ "${D}" == "8" ]]; then
        standard_run=izar_7days_stage.run
    else
        standard_run=izar_3days_stage.run
    fi
    for signed_h in -0.01 +0.01; do
        submit_chain "${D}" "${signed_h}" main "${standard_run}" \
            0.25 0.24 0.23 0.22 0.21 0.20 0.19 0.18 0.17 0.16
    done
    for signed_h in -0.02 +0.02; do
        submit_chain "${D}" "${signed_h}" main "${standard_run}" \
            0.24 0.22 0.20 0.18 0.16 0.14 0.12 0.10
    done
    for signed_h in -0.04 +0.04; do
        submit_chain "${D}" "${signed_h}" main "${standard_run}" \
            0.24 0.22 0.20 0.18 0.16 0.14 0.12 0.10
    done
    for signed_h in -0.08 +0.08; do
        submit_chain "${D}" "${signed_h}" main "${standard_run}" \
            0.23 0.20 0.17 0.14 0.11 0.08 0.05 0.02
    done
    for signed_h in -0.08 +0.08; do
        submit_chain "${D}" "${signed_h}" extra izar_7days_stage.run \
            0.24 0.22 0.20 0.18 0.16 0.14 0.12
    done
done

[[ "${job_count}" == "246" && "${head_count}" == "30" && "${dependency_count}" == "216" ]] || {
    echo "Internal count error: jobs=${job_count}, heads=${head_count}, dependencies=${dependency_count}" >&2
    exit 6
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 246 jobs = 30 heads + 216 afterok jobs; nothing submitted."
else
    echo "Submitted 246 jobs = 30 heads + 216 afterok jobs."
fi
