#!/usr/bin/env bash
# Sixteen Kuma heads.  Each head is one fixed-(D,signed h) adiabatic chain,
# and every later J2 stage resumes the immediately preceding optimized tensor.
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

for required in main_C3_LBFGS.py core_C3.py run_one_stage.sh kuma_stage.run; do
    [[ -s "${BUNDLE_DIR}/${required}" ]] || { echo "Missing ${required}" >&2; exit 3; }
done
LOCKED_CHI='[ 36, 54, 72, 90,108,126,144,162,160,160,180]'
grep -Fq "CHI_MIN_LIST  = ${LOCKED_CHI}" main_C3_LBFGS.py || {
    echo "Bundled main_C3_LBFGS.py does not have the locked CHI_MIN_LIST" >&2; exit 3;
}
grep -Fq "CHI_MAX_LIST  = ${LOCKED_CHI}" main_C3_LBFGS.py || {
    echo "Bundled main_C3_LBFGS.py does not have the locked CHI_MAX_LIST" >&2; exit 3;
}
for D in 8 9; do
    for texture in dimer plaquette; do
        for field_code in 0p01 0p02 0p04; do
            [[ -s "${BUNDLE_DIR}/seeds/D${D}/${texture}/h_${field_code}/J2_0p26.pt" ]] || {
                echo "Missing D=${D} ${texture} h=${field_code}, J2=0.26 seed" >&2; exit 3;
            }
        done
        [[ -s "${BUNDLE_DIR}/seeds/D${D}/${texture}/h_0p08/J2_0p14.pt" ]] || {
            echo "Missing D=${D} ${texture} h=0p08, J2=0.14 seed" >&2; exit 3;
        }
    done
done
mkdir -p logs Results_Kuma_lowJ2_extended

job_count=0
head_count=0
dependency_count=0
LAST_JOB_ID=""

submit_one() {
    local job_name="$1" predecessor="$2" exports="$3" result job_id
    job_count=$((job_count + 1))
    if [[ -z "${predecessor}" ]]; then
        head_count=$((head_count + 1))
        printf '%03d HEAD    %s\n' "${job_count}" "${job_name}"
    else
        dependency_count=$((dependency_count + 1))
        printf '%03d AFTEROK %-12s <- %s\n' "${job_count}" "${job_name}" "${predecessor}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        job_id="dry${job_count}"
    else
        local -a args=(--parsable --chdir="${BUNDLE_DIR}" --job-name="${job_name}" --export="${exports}")
        [[ -n "${predecessor}" ]] && args+=(--dependency="afterok:${predecessor}")
        result="$(sbatch "${args[@]}" "${BUNDLE_DIR}/kuma_stage.run")"
        job_id="${result%%;*}"
    fi
    [[ -n "${job_id}" ]] || { echo "Could not parse sbatch job ID" >&2; exit 5; }
    LAST_JOB_ID="${job_id}"
}

submit_chain() {
    local D="$1" signed_h="$2" seed_j2="$3"
    shift 3
    local CHI ORIENTATION texture branch sign_code sign_label sign_seed
    local field field_code seed_j2_code
    case "${D}" in
        8) CHI=144; ORIENTATION=1 ;;
        9) CHI=162; ORIENTATION=1 ;;
        *) echo "Unsupported D=${D}" >&2; exit 2 ;;
    esac
    if [[ "${signed_h}" == -* ]]; then
        texture=dimer; branch=dimer-plaquette; sign_code=d; sign_label=m; sign_seed=0
    else
        texture=plaquette; branch=plaquette; sign_code=p; sign_label=p; sign_seed=1
    fi
    field="${signed_h#[-+]}"
    field_code="${field/./p}"
    seed_j2_code="${seed_j2/./p}"

    local predecessor=""
    local input_rel="seeds/D${D}/${texture}/h_${field_code}/J2_${seed_j2_code}.pt"
    local j2 j2_code output_rel job_name field_seed j2_seed random_seed exports
    field_seed=$((10#${field#0.}))
    for j2 in "$@"; do
        j2_code="${j2/./p}"
        output_rel="Results_Kuma_lowJ2_extended/D_${D}/${texture}/h_${sign_label}${field_code}/J2_${j2_code}"
        job_name="e${D}${sign_code}${field_code#0p}${j2_code#0p}"
        j2_seed=$((10#${j2#0.}))
        random_seed=$((7200000 + 100000 * D + 50000 * sign_seed + 1000 * field_seed + j2_seed))
        exports="ALL,BUNDLE_DIR=${BUNDLE_DIR},D=${D},CHI=${CHI},ORIENTATION=${ORIENTATION},BRANCH=${branch},SIGNED_H=${signed_h},FIELD=${field},TARGET_J2=${j2},STAGE_HOURS=48,INPUT_REL=${input_rel},OUTPUT_REL=${output_rel},RANDOM_SEED=${random_seed}"
        submit_one "${job_name}" "${predecessor}" "${exports}"
        predecessor="${LAST_JOB_ID}"
        input_rel="${output_rel}/sweep_D${D}_chi${CHI}_best.pt"
    done
}

for D in 8 9; do
    # These four chains start from the already optimized Kuma J2=0.14 states.
    submit_chain "${D}" -0.08 0.14 0.12 0.10 0.08 0.06
    submit_chain "${D}" +0.08 0.14 0.12 0.10 0.08 0.06

    # The |h|=0.04 chains continue farther than the |h|=0.02 chains.
    for signed_h in -0.04 +0.04; do
        submit_chain "${D}" "${signed_h}" 0.26 \
            0.24 0.22 0.20 0.18 0.16 0.14 0.12
    done
    for signed_h in -0.02 +0.02; do
        submit_chain "${D}" "${signed_h}" 0.26 \
            0.24 0.22 0.20 0.18 0.16
    done

    submit_chain "${D}" -0.01 0.26 0.24 0.22 0.21 0.20
    submit_chain "${D}" +0.01 0.26 0.24 0.22 0.21 0.20
done

[[ "${job_count}" == "80" && "${head_count}" == "16" && "${dependency_count}" == "64" ]] || {
    echo "Internal count error: jobs=${job_count}, heads=${head_count}, dependencies=${dependency_count}" >&2
    exit 6
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 80 jobs = 16 heads + 64 afterok jobs; nothing submitted."
else
    echo "Submitted 80 jobs = 16 heads + 64 afterok jobs."
fi

