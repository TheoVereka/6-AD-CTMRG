#!/bin/bash
# Submit one Slurm job per selected manifest checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_NAME="${BUNDLE_NAME:-$(basename "${SCRIPT_DIR}")}"
SCRATCH_BUNDLE_DIR="/scratch/chye/${BUNDLE_NAME}"
HOME_BUNDLE_DIR="/home/chye/${BUNDLE_NAME}"
MANIFEST="${SCRATCH_BUNDLE_DIR}/checkpoint_manifest.tsv"
RUN_FILE="${SCRATCH_BUNDLE_DIR}/correlation_length_job.run"

D_SPEC="7 8 9 10"
D_SPEC_SET=0
MIN_D=""
J2_SPEC="all"
DRY_RUN=0
RESUBMIT_VALID=0

usage() {
    echo "Usage: $0 [--J2 all|0.24,0.25|J2_0p24,...] [--D 7,8,9,10]"
    echo "          [--min-D 7] [--dry-run] [--resubmit-valid]"
    echo "--min-D discovers every manifest D at or above the threshold."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --J2)
            J2_SPEC="${2:?--J2 requires a comma/space-separated value}"
            shift 2
            ;;
        --D)
            D_SPEC="${2:?--D requires a comma/space-separated value}"
            D_SPEC_SET=1
            shift 2
            ;;
        --min-D)
            MIN_D="${2:?--min-D requires an integer}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --resubmit-valid)
            RESUBMIT_VALID=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "${MANIFEST}" ]] || {
    echo "Missing manifest: ${MANIFEST}" >&2
    exit 2
}
[[ -f "${RUN_FILE}" ]] || {
    echo "Missing Slurm run file: ${RUN_FILE}" >&2
    exit 2
}
[[ -f "${HOME_BUNDLE_DIR}/run_one_correlation_length.py" ]] || {
    echo "Missing worker in home copy: ${HOME_BUNDLE_DIR}" >&2
    exit 2
}

if [[ "${D_SPEC_SET}" -eq 1 && -n "${MIN_D}" ]]; then
    echo "--D and --min-D are mutually exclusive." >&2
    exit 2
fi
if [[ -n "${MIN_D}" && ! "${MIN_D}" =~ ^[0-9]+$ ]]; then
    echo "--min-D must be a non-negative integer: ${MIN_D}" >&2
    exit 2
fi

declare -A STAGED_BY_KEY
declare -A J2_BY_TOKEN
declare -A TOKEN_BY_VALUE
declare -a ALL_TOKENS
declare -a ALL_DS

while IFS=$'\t' read -r TOKEN J2_VALUE D_BOND STAGED ORIGINAL SHA256; do
    [[ "${TOKEN}" == "j2_directory" ]] && continue
    KEY="${TOKEN}|${D_BOND}"
    STAGED_BY_KEY["${KEY}"]="${STAGED}"
    J2_BY_TOKEN["${TOKEN}"]="${J2_VALUE}"
    TOKEN_BY_VALUE["${J2_VALUE}"]="${TOKEN}"
    if [[ " ${ALL_TOKENS[*]-} " != *" ${TOKEN} "* ]]; then
        ALL_TOKENS+=("${TOKEN}")
    fi
    if [[ " ${ALL_DS[*]-} " != *" ${D_BOND} "* ]]; then
        ALL_DS+=("${D_BOND}")
    fi
done < "${MANIFEST}"

declare -a SELECTED_DS
if [[ -n "${MIN_D}" ]]; then
    for D_BOND in "${ALL_DS[@]}"; do
        if (( D_BOND >= MIN_D )); then
            SELECTED_DS+=("${D_BOND}")
        fi
    done
    if [[ "${#SELECTED_DS[@]}" -eq 0 ]]; then
        echo "No manifest checkpoint has D >= ${MIN_D}." >&2
        exit 2
    fi
else
    D_SPEC="${D_SPEC//,/ }"
    read -r -a SELECTED_DS <<< "${D_SPEC}"
fi

declare -a SELECTED_TOKENS
if [[ "${J2_SPEC}" == "all" ]]; then
    SELECTED_TOKENS=("${ALL_TOKENS[@]}")
else
    J2_SPEC="${J2_SPEC//,/ }"
    read -r -a REQUESTED_J2 <<< "${J2_SPEC}"
    for VALUE in "${REQUESTED_J2[@]}"; do
        if [[ -n "${J2_BY_TOKEN[${VALUE}]+x}" ]]; then
            SELECTED_TOKENS+=("${VALUE}")
        elif [[ -n "${TOKEN_BY_VALUE[${VALUE}]+x}" ]]; then
            SELECTED_TOKENS+=("${TOKEN_BY_VALUE[${VALUE}]}")
        else
            echo "Requested J2 is absent from the manifest: ${VALUE}" >&2
            exit 2
        fi
    done
fi

submitted=0
skipped=0
missing=0
for TOKEN in "${SELECTED_TOKENS[@]}"; do
    for D_BOND in "${SELECTED_DS[@]}"; do
        KEY="${TOKEN}|${D_BOND}"
        if [[ -z "${STAGED_BY_KEY[${KEY}]+x}" ]]; then
            echo "MISSING checkpoint for ${TOKEN}, D=${D_BOND}"
            missing=$((missing + 1))
            continue
        fi

        CHECKPOINT="${SCRATCH_BUNDLE_DIR}/checkpoints/${STAGED_BY_KEY[${KEY}]}"
        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "MISSING staged file: ${CHECKPOINT}"
            missing=$((missing + 1))
            continue
        fi

        if [[ "${RESUBMIT_VALID}" -eq 0 ]] && \
           python "${HOME_BUNDLE_DIR}/run_one_correlation_length.py" \
               --bundle-root "${SCRATCH_BUNDLE_DIR}" \
               --check-only \
               "${TOKEN}" "${D_BOND}"; then
            echo "SKIP valid existing result for ${TOKEN}, D=${D_BOND}"
            skipped=$((skipped + 1))
            continue
        fi

        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "WOULD SUBMIT ${TOKEN}, D=${D_BOND}"
        else
            (
                cd "${SCRATCH_BUNDLE_DIR}"
                sbatch \
                    --job-name="cl-${TOKEN#J2_}-D${D_BOND}" \
                    "${RUN_FILE}" "${TOKEN}" "${D_BOND}" "${BUNDLE_NAME}"
            )
        fi
        submitted=$((submitted + 1))
    done
done

echo "Submission summary: selected=${submitted}, skipped=${skipped}, missing=${missing}, dry_run=${DRY_RUN}."
