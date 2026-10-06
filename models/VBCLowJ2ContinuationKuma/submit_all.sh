#!/usr/bin/env bash
# Four Kuma heads: D=8,9 and signed h=-0.08,+0.08.  Every head owns an
# afterok chain J2=0.24,0.22,...,0.14.
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
        [[ -s "${BUNDLE_DIR}/seeds/D${D}/${texture}/h_0p08.pt" ]] || {
            echo "Missing D=${D} ${texture} seed" >&2; exit 3;
        }
    done
done
mkdir -p logs Results_Kuma_lowJ2

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
    local D="$1" signed_h="$2" CHI ORIENTATION texture branch sign_code sign_label sign_seed
    shift 2
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
    local predecessor="" input_rel="seeds/D${D}/${texture}/h_0p08.pt"
    local j2 j2_code output_rel job_name random_seed exports
    for j2 in "$@"; do
        j2_code="${j2/./p}"
        output_rel="Results_Kuma_lowJ2/D_${D}/${texture}/h_${sign_label}0p08/J2_${j2_code}"
        job_name="k${D}${sign_code}08${j2_code#0p}"
        random_seed=$((6800000 + 10000 * D + 1000 * sign_seed + 10#${j2#0.}))
        exports="ALL,BUNDLE_DIR=${BUNDLE_DIR},D=${D},CHI=${CHI},ORIENTATION=${ORIENTATION},BRANCH=${branch},SIGNED_H=${signed_h},FIELD=0.08,TARGET_J2=${j2},STAGE_HOURS=48,INPUT_REL=${input_rel},OUTPUT_REL=${output_rel},RANDOM_SEED=${random_seed}"
        submit_one "${job_name}" "${predecessor}" "${exports}"
        predecessor="${LAST_JOB_ID}"
        input_rel="${output_rel}/sweep_D${D}_chi${CHI}_best.pt"
    done
}

for D in 8 9; do
    submit_chain "${D}" -0.08 0.24 0.22 0.20 0.18 0.16 0.14
    submit_chain "${D}" +0.08 0.24 0.22 0.20 0.18 0.16 0.14
done

[[ "${job_count}" == "24" && "${head_count}" == "4" && "${dependency_count}" == "20" ]] || {
    echo "Internal count error: jobs=${job_count}, heads=${head_count}, dependencies=${dependency_count}" >&2
    exit 6
}
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: 24 jobs = 4 heads + 20 afterok jobs; nothing submitted."
else
    echo "Submitted 24 jobs = 4 heads + 20 afterok jobs."
fi
