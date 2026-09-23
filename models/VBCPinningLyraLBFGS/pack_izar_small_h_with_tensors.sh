#!/usr/bin/env bash
# Pack every completed small-h stage together with its optimized best tensor.
#
# Run from ~/VBCPinningLyraLBFGS on Izar.  Unlike the earlier observation-only
# archive, this archive is sufficient for zero-field tensor selection.

set -euo pipefail

RESULT_ROOT="${1:-Results_Izar_replica1}"
ARCHIVE="${2:-Izar_small_h_completed_with_tensors.tar.gz}"

if [[ ! -d "${RESULT_ROOT}" ]]; then
    echo "ERROR: result root does not exist: ${RESULT_ROOT}" >&2
    exit 2
fi

STAGE_LIST="$(mktemp)"
FILE_LIST="$(mktemp)"
trap 'rm -f "${STAGE_LIST}" "${FILE_LIST}"' EXIT

# The base observation is written only after optimization/evaluation finishes.
# Requiring it to be at least one minute old avoids racing a final filesystem
# flush while jobs are still running.  Lookahead observations are not used as
# independent stages.
find "${RESULT_ROOT}" -type f \
    -name 'D_*_chi_*_energy_magnetization_correlation.txt' \
    ! -name '*_lookahead_*' -mmin +1 -print0 |
while IFS= read -r -d '' OBS; do
    STAGE="$(dirname "${OBS}")"
    FIELD="$(basename "${STAGE}")"
    case "${FIELD}" in
        h_0p005|h_0p003|h_0p002|h_0p001|h_0) ;;
        *) continue ;;
    esac

    BASE="$(basename "${OBS}")"
    if [[ ! "${BASE}" =~ ^D_([0-9]+)_chi_([0-9]+)_energy_magnetization_correlation\.txt$ ]]; then
        echo "WARNING: unrecognized observation name: ${OBS}" >&2
        continue
    fi
    D="${BASH_REMATCH[1]}"
    CHI="${BASH_REMATCH[2]}"
    HYPERPARAMS="${STAGE}/hyperparams.yaml"
    TENSOR="${STAGE}/sweep_D${D}_chi${CHI}_best.pt"

    if [[ ! -s "${HYPERPARAMS}" ]]; then
        echo "WARNING: missing/nonempty hyperparams: ${HYPERPARAMS}" >&2
        continue
    fi
    if [[ ! -s "${TENSOR}" ]]; then
        echo "WARNING: missing/nonempty optimized tensor: ${TENSOR}" >&2
        continue
    fi

    printf '%s\0' "${OBS}" "${HYPERPARAMS}" "${TENSOR}" >> "${FILE_LIST}"
    while IFS= read -r -d '' LOOKAHEAD; do
        printf '%s\0' "${LOOKAHEAD}" >> "${FILE_LIST}"
    done < <(find "${STAGE}" -maxdepth 1 -type f \
        -name "D_${D}_chi_${CHI}_lookahead_*_energy_magnetization_correlation.txt" \
        -print0)
    printf '%s\0' "${STAGE}" >> "${STAGE_LIST}"
done

if [[ ! -s "${FILE_LIST}" ]]; then
    echo "ERROR: no completed small-h stages with best tensors found" >&2
    exit 3
fi

# One path exactly once.  GNU tar receives NUL-delimited names, so no quoting
# or whitespace in a path can alter the archive contents.
sort -zu "${FILE_LIST}" -o "${FILE_LIST}"
sort -zu "${STAGE_LIST}" -o "${STAGE_LIST}"
tar --null --verbatim-files-from -czf "${ARCHIVE}" -T "${FILE_LIST}"

FILE_COUNT="$(tr -cd '\0' < "${FILE_LIST}" | wc -c)"
STAGE_COUNT="$(tr -cd '\0' < "${STAGE_LIST}" | wc -c)"
echo "Packed ${STAGE_COUNT} completed stages / ${FILE_COUNT} unique files"
du -h "${ARCHIVE}"

