#!/usr/bin/env bash
# Make an atomic snapshot containing only completed Izar low-J2 stages.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

RESULTS_ROOT="Results_Izar_lowJ2"
ARCHIVE="Izar_lowJ2_completed.tar.gz"
MANIFEST="Izar_lowJ2_completed_manifest.tsv"
TMP_LIST="$(mktemp)"
TMP_ARCHIVE="${ARCHIVE}.tmp.$$"
trap 'rm -f -- "${TMP_LIST}" "${TMP_ARCHIVE}"' EXIT

printf 'stage\tD\tchi\tbytes_observation\tbytes_tensor\n' > "${MANIFEST}"
count=0

if [[ -d "${RESULTS_ROOT}" ]]; then
    while IFS= read -r -d '' observation; do
        filename="$(basename -- "${observation}")"
        if [[ ! "${filename}" =~ ^D_([0-9]+)_chi_([0-9]+)_energy_magnetization_correlation\.txt$ ]]; then
            continue
        fi
        D="${BASH_REMATCH[1]}"
        chi="${BASH_REMATCH[2]}"
        # D=8 from Izar is permanently banned from this low-J2 project.
        [[ "${D}" == "8" ]] && continue
        stage="$(dirname -- "${observation}")"
        tensor="${stage}/sweep_D${D}_chi${chi}_best.pt"
        summary="${stage}/sweep_results.json"
        hyperparams="${stage}/hyperparams.yaml"
        if [[ ! -s "${observation}" || ! -s "${tensor}" \
              || ! -s "${summary}" || ! -s "${hyperparams}" ]]; then
            continue
        fi
        printf '%s\0' "${stage}" >> "${TMP_LIST}"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${stage}" "${D}" "${chi}" \
            "$(stat -c %s -- "${observation}")" \
            "$(stat -c %s -- "${tensor}")" >> "${MANIFEST}"
        count=$((count + 1))
    done < <(find "${RESULTS_ROOT}" -type f \
        -name 'D_*_chi_*_energy_magnetization_correlation.txt' -print0)
fi

# Each leaf stage occurs only once, but sorting makes the archive deterministic.
sort -zu -o "${TMP_LIST}" "${TMP_LIST}"
printf '%s\0' "${MANIFEST}" >> "${TMP_LIST}"
tar --null --files-from="${TMP_LIST}" -czf "${TMP_ARCHIVE}"
mv -f -- "${TMP_ARCHIVE}" "${ARCHIVE}"

echo "Packed ${count} completed stages into ${BUNDLE_DIR}/${ARCHIVE}"
du -h -- "${ARCHIVE}"

