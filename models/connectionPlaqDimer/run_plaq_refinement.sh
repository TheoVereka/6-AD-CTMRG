#!/usr/bin/env bash
# One three-day LBFGS stage, starting from the original 62.5% F/R tensor or
# from the previous percentage in the same independent continuation.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
ORIGINAL_BUNDLE="${ORIGINAL_BUNDLE:-/scratch/izar/chye/connectionPlaqDimer_trees_20261003}"
RESULTS_DIR="${BUNDLE_DIR}/results_plaq_refinement_20261006"

select_best_checkpoint() {
    local directory="$1" D="$2" path name chi best_chi=-1 best_path=""
    local -a matches=()
    shopt -s nullglob
    matches=("${directory}"/sweep_D"${D}"_chi*_best.pt)
    shopt -u nullglob
    for path in "${matches[@]}"; do
        [[ -s "${path}" ]] || continue
        name="${path##*/}"
        if [[ "${name}" =~ ^sweep_D${D}_chi([0-9]+)_best\.pt$ ]]; then
            chi="${BASH_REMATCH[1]}"
            if (( 10#${chi} > best_chi )); then
                best_chi=$((10#${chi}))
                best_path="${path}"
            fi
        fi
    done
    [[ -n "${best_path}" ]] || {
        echo "No best checkpoint for D=${D} in ${directory}" >&2
        return 1
    }
    printf '%s\n' "${best_path}"
}

checkpoint_from_marker() {
    local directory="$1" D="$2" marker checkpoint name
    marker="${directory}/COMPLETED.stage"
    [[ -s "${marker}" ]] || { echo "Missing completion marker: ${marker}" >&2; return 1; }
    checkpoint="$(sed -n 's/^checkpoint=//p' "${marker}")"
    [[ "${checkpoint}" == "${directory}/"* && -s "${checkpoint}" ]] || {
        echo "Checkpoint recorded in ${marker} is missing or outside its stage" >&2
        return 1
    }
    name="${checkpoint##*/}"
    [[ "${name}" =~ ^sweep_D${D}_chi[0-9]+_best\.pt$ ]] || {
        echo "Wrong-D checkpoint recorded in ${marker}: ${checkpoint}" >&2
        return 1
    }
    printf '%s\n' "${checkpoint}"
}

check_original_source() {
    local D="$1" source_node="$2" directory checkpoint marker expected_chi
    directory="${ORIGINAL_BUNDLE}/results/D${D}_plaq/${source_node}"
    marker="${directory}/COMPLETED.stage"
    [[ -s "${marker}" ]] || { echo "Missing source completion marker: ${marker}" >&2; return 1; }
    for expected in "D=${D}" 'connection=plaq' 't=5'; do
        grep -qx "${expected}" "${marker}" || {
            echo "Source marker mismatch (${expected}): ${marker}" >&2
            return 1
        }
    done
    checkpoint="$(checkpoint_from_marker "${directory}" "${D}")"
    if [[ "${D}" == 6 ]]; then expected_chi=108; else expected_chi=126; fi
    [[ "${checkpoint##*/}" == "sweep_D${D}_chi${expected_chi}_best.pt" ]] || {
        echo "Unexpected source chi in ${marker}: ${checkpoint}" >&2
        return 1
    }
    printf 'Source D=%s %s: %s\n' "${D}" "${source_node}" "${checkpoint}" >&2
}

if [[ "${1:-}" == --check-sources && $# -eq 1 ]]; then
    for D in 6 7; do
        for source_node in F R; do
            check_original_source "${D}" "${source_node}"
        done
    done
    exit 0
fi
[[ $# -eq 0 ]] || { echo 'Usage: bash run_plaq_refinement.sh [--check-sources]' >&2; exit 2; }

: "${D:?D is required}"
: "${SOURCE_NODE:?SOURCE_NODE is required}"
: "${REPLICA:?REPLICA is required}"
: "${PERCENT:?PERCENT is required}"
: "${PREV_PERCENT:?PREV_PERCENT is required}"
: "${RUN_SEED:?RUN_SEED is required}"
[[ "${D}" == 6 || "${D}" == 7 ]] || { echo "Invalid D=${D}" >&2; exit 2; }
if [[ "${D}" == 6 ]]; then CHI=108; else CHI=126; fi
[[ "${SOURCE_NODE}" == F || "${SOURCE_NODE}" == R ]] || {
    echo "Invalid SOURCE_NODE=${SOURCE_NODE}" >&2; exit 2;
}
[[ "${REPLICA}" =~ ^0[1-4]$ ]] || { echo "Invalid REPLICA=${REPLICA}" >&2; exit 2; }
[[ "${RUN_SEED}" =~ ^[0-9]+$ ]] || { echo "Invalid RUN_SEED=${RUN_SEED}" >&2; exit 2; }

case "${PERCENT}" in
    70) T=5.6;    EXPECTED_PREV=none ;;
    75) T=6;      EXPECTED_PREV=70 ;;
    80) T=6.4;    EXPECTED_PREV=75 ;;
    84) T=6.72;   EXPECTED_PREV=80 ;;
    88) T=7.04;   EXPECTED_PREV=84 ;;
    91) T=7.28;   EXPECTED_PREV=88 ;;
    94) T=7.52;   EXPECTED_PREV=91 ;;
    96) T=7.68;   EXPECTED_PREV=94 ;;
    98) T=7.84;   EXPECTED_PREV=96 ;;
    99) T=7.92;   EXPECTED_PREV=98 ;;
    100) T=8;     EXPECTED_PREV=99 ;;
    *) echo "Invalid PERCENT=${PERCENT}" >&2; exit 2 ;;
esac
[[ "${PREV_PERCENT}" == "${EXPECTED_PREV}" ]] || {
    echo "Wrong predecessor ${PREV_PERCENT} for ${PERCENT}%" >&2
    exit 2
}

SEQUENCE_DIR="${RESULTS_DIR}/D${D}_${SOURCE_NODE}/rep${REPLICA}"
OUTPUT_DIR="${SEQUENCE_DIR}/p${PERCENT}"
[[ ! -e "${OUTPUT_DIR}/COMPLETED.stage" ]] || {
    echo "Stage already completed: ${OUTPUT_DIR}" >&2
    exit 3
}
if [[ "${PREV_PERCENT}" == none ]]; then
    check_original_source "${D}" "${SOURCE_NODE}"
    PREV_DIR="${ORIGINAL_BUNDLE}/results/D${D}_plaq/${SOURCE_NODE}"
else
    PREV_DIR="${SEQUENCE_DIR}/p${PREV_PERCENT}"
    [[ -s "${PREV_DIR}/COMPLETED.stage" ]] || {
        echo "Previous stage has no completion marker: ${PREV_DIR}" >&2
        exit 3
    }
fi
RESUME_CKPT="$(checkpoint_from_marker "${PREV_DIR}" "${D}")"
mkdir -p "${OUTPUT_DIR}"

echo "D=${D} source=${SOURCE_NODE} duplicate=${REPLICA} percentage=${PERCENT} t=${T}"
echo "Resume=${RESUME_CKPT}; seed=${RUN_SEED}; output=${OUTPUT_DIR}"
python -u "${BUNDLE_DIR}/main_C3_LBFGS.py" \
    --connection plaq --t "${T}" --ansatz twoc3 --Ds "${D}" \
    --chi-min "${CHI}" --chi-max "${CHI}" --chi-step 1 \
    --hours 70 --gpu --ngpu 1 --optimizer lbfgs \
    --no-mean-field-init --resume "${RESUME_CKPT}" --resume-tensors-only \
    --fix-seed --seed "${RUN_SEED}" --output-dir "${OUTPUT_DIR}"

[[ -s "${OUTPUT_DIR}/sweep_results.json" ]] || {
    echo "Run returned without sweep_results.json: ${OUTPUT_DIR}" >&2
    exit 4
}
BEST_CKPT="$(select_best_checkpoint "${OUTPUT_DIR}" "${D}")"
printf 'D=%s\nsource_node=%s\nreplica=%s\npercent=%s\nt=%s\nseed=%s\ncheckpoint=%s\ncompleted_utc=%s\n' \
    "${D}" "${SOURCE_NODE}" "${REPLICA}" "${PERCENT}" "${T}" "${RUN_SEED}" \
    "${BEST_CKPT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${OUTPUT_DIR}/COMPLETED.stage.tmp"
mv -f "${OUTPUT_DIR}/COMPLETED.stage.tmp" "${OUTPUT_DIR}/COMPLETED.stage"
echo "COMPLETED ${OUTPUT_DIR}: ${BEST_CKPT}"
