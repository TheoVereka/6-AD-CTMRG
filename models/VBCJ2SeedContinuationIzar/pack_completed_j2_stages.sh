#!/usr/bin/env bash
# Snapshot every individually completed J2 stage, even when its chain continues.
set -euo pipefail

RESULT_ROOT="${1:-Results_Izar_J2_sequences}"
ARCHIVE="${2:-Izar_completed_J2_continuation.tar.gz}"
[[ -d "${RESULT_ROOT}" ]] || {
    echo "ERROR: result root does not exist: ${RESULT_ROOT}" >&2
    exit 2
}

STAGE_LIST="$(mktemp)"
FILE_LIST="$(mktemp)"
trap 'rm -f "${STAGE_LIST}" "${FILE_LIST}"' EXIT

# sweep_results.json is written only after optimization, observables, and the
# chi-lookahead have all returned. Requiring a quiet minute avoids racing the
# final filesystem flush of a just-finished Slurm stage.
while IFS= read -r -d '' RESULTS_JSON; do
    STAGE="$(dirname "${RESULTS_JSON}")"
    mapfile -d '' -t OBSERVATIONS < <(
        find "${STAGE}" -maxdepth 1 -type f \
            -name 'D_*_chi_*_energy_magnetization_correlation.txt' \
            ! -name '*_lookahead_*' -print0
    )
    if (( ${#OBSERVATIONS[@]} != 1 )); then
        echo "WARNING: completed stage has ${#OBSERVATIONS[@]} base observations: ${STAGE}" >&2
        continue
    fi
    OBS="${OBSERVATIONS[0]}"
    BASE="$(basename "${OBS}")"
    if [[ ! "${BASE}" =~ ^D_([0-9]+)_chi_([0-9]+)_energy_magnetization_correlation\.txt$ ]]; then
        echo "WARNING: unrecognized observation name: ${OBS}" >&2
        continue
    fi
    D="${BASH_REMATCH[1]}"
    CHI="${BASH_REMATCH[2]}"
    BEST="${STAGE}/sweep_D${D}_chi${CHI}_best.pt"
    HYPERPARAMS="${STAGE}/hyperparams.yaml"
    if [[ ! -s "${BEST}" || ! -s "${HYPERPARAMS}" ]]; then
        echo "WARNING: completed stage lacks best tensor/hyperparams: ${STAGE}" >&2
        continue
    fi
    printf '%s\0' "${STAGE}" >> "${STAGE_LIST}"
    while IFS= read -r -d '' FILE; do
        printf '%s\0' "${FILE}" >> "${FILE_LIST}"
    done < <(find "${STAGE}" -maxdepth 1 -type f -print0)
done < <(find "${RESULT_ROOT}" -type f -name 'sweep_results.json' -mmin +1 -print0)

if [[ ! -s "${STAGE_LIST}" ]]; then
    echo "ERROR: no individually completed J2 stages found" >&2
    exit 3
fi

for METADATA in selected_seed_manifest.csv submission_plan.tsv; do
    [[ -s "${METADATA}" ]] && printf '%s\0' "${METADATA}" >> "${FILE_LIST}"
done
sort -zu "${STAGE_LIST}" -o "${STAGE_LIST}"
sort -zu "${FILE_LIST}" -o "${FILE_LIST}"
tar --null --verbatim-files-from -czf "${ARCHIVE}" -T "${FILE_LIST}"

STAGE_COUNT="$(tr -cd '\0' < "${STAGE_LIST}" | wc -c)"
FILE_COUNT="$(tr -cd '\0' < "${FILE_LIST}" | wc -c)"
echo "Packed ${STAGE_COUNT} individually completed J2 stages / ${FILE_COUNT} files"
du -h "${ARCHIVE}"

