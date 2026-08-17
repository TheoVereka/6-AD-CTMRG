#!/bin/bash
# Submit one Slurm job per selected manifest checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/checkpoint_manifest.tsv"
RUN_FILE="${SCRIPT_DIR}/correlation_length_job.run"
ORDINARY_JOB_PREFIX="clo7"

D_SPEC=""
D_SPEC_SET=0
MIN_D="3"
J2_SPEC="all"
ANSATZ_SPEC="all"
DRY_RUN=0
RESUBMIT_VALID=0

usage() {
    echo "Usage: $0 [--ansatz all|2tensor_twoC3,...] [--J2 all|0.24,0.25]"
    echo "          [--D 7,8,9,10]"
    echo "          [--min-D 7] [--dry-run] [--resubmit-valid]"
    echo "With no arguments, submit every available manifest case with D>=3."
    echo "--min-D discovers every manifest D at or above the threshold."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ansatz)
            ANSATZ_SPEC="${2:?--ansatz requires a comma/space-separated value}"
            shift 2
            ;;
        --J2)
            J2_SPEC="${2:?--J2 requires a comma/space-separated value}"
            shift 2
            ;;
        --D)
            D_SPEC="${2:?--D requires a comma/space-separated value}"
            D_SPEC_SET=1
            MIN_D=""
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
[[ -f "${SCRIPT_DIR}/run_one_correlation_length.py" ]] || {
    echo "Missing worker in bundle: ${SCRIPT_DIR}" >&2
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
declare -A HASH_BY_KEY
declare -A J2_BY_TOKEN
declare -A TOKEN_BY_VALUE
declare -A KNOWN_ANSATZ
declare -a ALL_TOKENS
declare -a ALL_ANSATZ
declare -a ALL_DS
declare -a ALL_KEYS

while IFS=$'\t' read -r TOKEN ANSATZ J2_VALUE D_BOND STAGED ORIGINAL SHA256; do
    [[ "${TOKEN}" == "j2_directory" ]] && continue
    if [[ -z "${ANSATZ}" || -z "${D_BOND}" || -z "${STAGED}" ]]; then
        echo "Malformed v2 manifest row. Rerun collect_checkpoints.py locally: ${TOKEN}" >&2
        exit 2
    fi
    KEY="${ANSATZ}|${TOKEN}|${D_BOND}"
    STAGED_BY_KEY["${KEY}"]="${STAGED}"
    HASH_BY_KEY["${KEY}"]="${SHA256}"
    ALL_KEYS+=("${KEY}")
    J2_BY_TOKEN["${TOKEN}"]="${J2_VALUE}"
    TOKEN_BY_VALUE["${J2_VALUE}"]="${TOKEN}"
    KNOWN_ANSATZ["${ANSATZ}"]=1
    if [[ " ${ALL_TOKENS[*]-} " != *" ${TOKEN} "* ]]; then
        ALL_TOKENS+=("${TOKEN}")
    fi
    if [[ " ${ALL_ANSATZ[*]-} " != *" ${ANSATZ} "* ]]; then
        ALL_ANSATZ+=("${ANSATZ}")
    fi
    if [[ " ${ALL_DS[*]-} " != *" ${D_BOND} "* ]]; then
        ALL_DS+=("${D_BOND}")
    fi
done < "${MANIFEST}"

if [[ "${#ALL_KEYS[@]}" -eq 0 ]]; then
    echo "No stale or missing current-tensor correlation lengths are present in the submit manifest."
    exit 0
fi

# Snapshot every active Slurm job for this user before submitting.  The new
# ordinary-v6 jobs have an explicit schema/ansatz prefix.  The legacy generic
# prefix is also treated as an active claim during migration because an older
# running batch may already be computing the same ordinary result.
declare -A ACTIVE_JOB_BY_NAME
SLURM_USER="${USER:-$(id -un)}"
if ! SQUEUE_OUTPUT="$(
    squeue --noheader --user="${SLURM_USER}" --format="%i|%.128j|%T"
)"; then
    echo "Failed to query active Slurm jobs with squeue; refusing to submit duplicates." >&2
    exit 2
fi
while IFS='|' read -r JOB_ID JOB_NAME JOB_STATE; do
    [[ -z "${JOB_ID}" ]] && continue
    JOB_ID="${JOB_ID#"${JOB_ID%%[![:space:]]*}"}"
    JOB_ID="${JOB_ID%"${JOB_ID##*[![:space:]]}"}"
    JOB_NAME="${JOB_NAME#"${JOB_NAME%%[![:space:]]*}"}"
    JOB_NAME="${JOB_NAME%"${JOB_NAME##*[![:space:]]}"}"
    JOB_STATE="${JOB_STATE#"${JOB_STATE%%[![:space:]]*}"}"
    JOB_STATE="${JOB_STATE%"${JOB_STATE##*[![:space:]]}"}"
    ACTIVE_JOB_BY_NAME["${JOB_NAME}"]="${JOB_ID}|${JOB_STATE}"
done <<< "${SQUEUE_OUTPUT}"

ansatz_job_token() {
    case "$1" in
        neel_free_param) echo "neelf" ;;
        neel_symmetrized) echo "neels" ;;
        1tensor_C6Ypi) echo "c6ypi" ;;
        1tensor_C3Vypi) echo "c3vypi" ;;
        2tensor_twoC3) echo "2c3" ;;
        *) echo "Unsupported C3 ansatz in manifest: $1" >&2; return 2 ;;
    esac
}

active_log_checkpoint_hash() {
    local job_id="$1"
    local log_file
    local match
    local -a logs=()
    shopt -s nullglob
    logs=("${SCRIPT_DIR}"/job-*-"${job_id}".out)
    shopt -u nullglob
    for log_file in "${logs[@]}"; do
        match="$(grep -Eo 'CHECKPOINT_SHA256=[0-9a-fA-F]{64}' "${log_file}" | tail -n 1 || true)"
        if [[ -n "${match}" ]]; then
            echo "${match#CHECKPOINT_SHA256=}"
            return 0
        fi
    done
    return 1
}

legacy_log_class() {
    local job_id="$1"
    local log_file
    local saw_generalized=0
    local -a logs=()
    shopt -s nullglob
    logs=("${SCRIPT_DIR}"/job-*-"${job_id}".out)
    shopt -u nullglob
    for log_file in "${logs[@]}"; do
        if grep -Eq 'compute_three_ordinary_correlation_lengths|DIAGONALIZE ordinary' "${log_file}"; then
            echo "ordinary"
            return 0
        fi
        if grep -Eq 'compute_three_generalized_correlation_lengths|DIAGONALIZE generalized' "${log_file}"; then
            saw_generalized=1
        fi
    done
    if [[ "${saw_generalized}" -eq 1 ]]; then
        echo "generalized"
    else
        echo "ambiguous"
    fi
}

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

declare -a SELECTED_ANSATZ
if [[ "${ANSATZ_SPEC}" == "all" ]]; then
    SELECTED_ANSATZ=("${ALL_ANSATZ[@]}")
else
    ANSATZ_SPEC="${ANSATZ_SPEC//,/ }"
    read -r -a SELECTED_ANSATZ <<< "${ANSATZ_SPEC}"
    for ANSATZ in "${SELECTED_ANSATZ[@]}"; do
        if [[ -z "${KNOWN_ANSATZ[${ANSATZ}]+x}" ]]; then
            echo "Requested ansatz is absent from the manifest: ${ANSATZ}" >&2
            exit 2
        fi
    done
fi

declare -A SELECTED_D_SET
declare -A SELECTED_TOKEN_SET
declare -A SELECTED_ANSATZ_SET
for D_BOND in "${SELECTED_DS[@]}"; do SELECTED_D_SET["${D_BOND}"]=1; done
for TOKEN in "${SELECTED_TOKENS[@]}"; do SELECTED_TOKEN_SET["${TOKEN}"]=1; done
for ANSATZ in "${SELECTED_ANSATZ[@]}"; do SELECTED_ANSATZ_SET["${ANSATZ}"]=1; done

submitted=0
skipped_completed=0
skipped_active=0
missing=0
for KEY in "${ALL_KEYS[@]}"; do
    IFS='|' read -r ANSATZ TOKEN D_BOND <<< "${KEY}"
    [[ -z "${SELECTED_ANSATZ_SET[${ANSATZ}]+x}" ]] && continue
    [[ -z "${SELECTED_TOKEN_SET[${TOKEN}]+x}" ]] && continue
    [[ -z "${SELECTED_D_SET[${D_BOND}]+x}" ]] && continue

    CHECKPOINT="${SCRIPT_DIR}/checkpoints/${STAGED_BY_KEY[${KEY}]}"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "MISSING staged file: ${CHECKPOINT}"
        missing=$((missing + 1))
        continue
    fi

    if [[ "${RESUBMIT_VALID}" -eq 0 ]] && \
       python "${SCRIPT_DIR}/run_one_correlation_length.py" \
           --bundle-root "${SCRIPT_DIR}" \
           --check-only \
           "${ANSATZ}" "${TOKEN}" "${D_BOND}"; then
        echo "SKIP result for identical checkpoint hash: ${ANSATZ}, ${TOKEN}, D=${D_BOND}"
        skipped_completed=$((skipped_completed + 1))
        continue
    fi

    ANSATZ_TOKEN="$(ansatz_job_token "${ANSATZ}")"
    JOB_SUFFIX="${TOKEN#J2_}-D${D_BOND}"
    CHECKPOINT_HASH="${HASH_BY_KEY[${KEY}]}"
    HASH_TOKEN="${CHECKPOINT_HASH:0:12}"
    JOB_NAME="${ORDINARY_JOB_PREFIX}-${ANSATZ_TOKEN}-${JOB_SUFFIX}-${HASH_TOKEN}"
    V6_JOB_NAME="clo6-${ANSATZ_TOKEN}-${JOB_SUFFIX}"
    V5_JOB_NAME="clo5-2c3-${JOB_SUFFIX}"
    LEGACY_NAME="cl-${JOB_SUFFIX}"
    ACTIVE_NAME=""
    if [[ -n "${ACTIVE_JOB_BY_NAME[${JOB_NAME}]+x}" ]]; then
        ACTIVE_NAME="${JOB_NAME}"
    else
        declare -a OLD_NAMES=("${V6_JOB_NAME}")
        if [[ "${ANSATZ}" == "2tensor_twoC3" ]]; then
            OLD_NAMES+=("${V5_JOB_NAME}" "${LEGACY_NAME}")
        fi
        for OLD_NAME in "${OLD_NAMES[@]}"; do
            [[ -z "${ACTIVE_JOB_BY_NAME[${OLD_NAME}]+x}" ]] && continue
            IFS='|' read -r OLD_ID OLD_STATE \
                <<< "${ACTIVE_JOB_BY_NAME[${OLD_NAME}]}"
            LOGGED_HASH="$(active_log_checkpoint_hash "${OLD_ID}" || true)"
            if [[ -n "${LOGGED_HASH}" && "${LOGGED_HASH}" != "${CHECKPOINT_HASH}" ]]; then
                echo "ACTIVE old job ${OLD_ID} (${OLD_NAME}) uses stale tensor ${LOGGED_HASH:0:12}; it does not block current tensor ${HASH_TOKEN}."
                continue
            fi
            ACTIVE_NAME="${OLD_NAME}"
            LEGACY_CLASS="$(legacy_log_class "${OLD_ID}")"
            if [[ -n "${LOGGED_HASH}" ]]; then
                echo "Old active job ${OLD_ID} matches current tensor ${HASH_TOKEN} and is classified as ${LEGACY_CLASS}."
            else
                echo "Old active job ${OLD_ID} has no logged tensor hash; treating it conservatively as a possible current-tensor claim until it leaves squeue."
            fi
            break
        done
    fi
    if [[ -n "${ACTIVE_NAME}" ]]; then
        IFS='|' read -r ACTIVE_ID ACTIVE_STATE \
            <<< "${ACTIVE_JOB_BY_NAME[${ACTIVE_NAME}]}"
        echo "SKIP active job ${ACTIVE_ID} (${ACTIVE_STATE}, ${ACTIVE_NAME}) for ${ANSATZ}, ${TOKEN}, D=${D_BOND}"
        skipped_active=$((skipped_active + 1))
        continue
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "WOULD SUBMIT ${ANSATZ}, ${TOKEN}, D=${D_BOND} as ${JOB_NAME}"
    else
        (
            cd "${SCRIPT_DIR}"
            sbatch \
                --dependency=singleton \
                --job-name="${JOB_NAME}" \
                "${RUN_FILE}" "${ANSATZ}" "${TOKEN}" "${D_BOND}" "${SCRIPT_DIR}"
        )
        ACTIVE_JOB_BY_NAME["${JOB_NAME}"]="submitted-now|PENDING"
    fi
    submitted=$((submitted + 1))
done

echo "Submission summary: submitted=${submitted}, skipped_completed=${skipped_completed}, skipped_active=${skipped_active}, missing=${missing}, dry_run=${DRY_RUN}."
